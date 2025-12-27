#!/usr/bin/env python3
"""
multi_gpu_gliner2_hf_dataset_infer.py

Multi-GPU entity extraction with GLiNER2 over a Hugging Face dataset.
One process per GPU (Accelerate), each processes a shard and writes its own JSONL part.

Examples:
  # From Hub
  accelerate launch --num_processes 4 multi_gpu_gliner2_hf_dataset_infer.py \
    --dataset cnn_dailymail --config 3.0.0 --split test \
    --text-field article \
    --labels person,company,location \
    --output out --batch-size 64 --bf16 --streaming

  # From disk (saved via datasets.save_to_disk)
  accelerate launch --num_processes 4 multi_gpu_gliner2_hf_dataset_infer.py \
    --dataset /path/to/dataset_on_disk --split train \
    --text-field text --id-field id \
    --labels person,org,location \
    --output out --batch-size 64 --fp16
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Union

import torch
from accelerate import PartialState
from gliner2 import GLiNER2

import datasets  # pip install datasets


LabelsType = Union[List[str], Dict[str, str]]  # list OR {label: description}


@dataclass
class Record:
    idx: int
    rid: Union[int, str]
    text: str
    raw: Optional[dict] = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="fastino/gliner2-base-v1", help="HF model id for GLiNER2")

    # Dataset inputs
    p.add_argument("--dataset", required=True, help="HF dataset name (hub) OR path to load_from_disk dir")
    p.add_argument("--config", default=None, help="HF dataset config name (if applicable)")
    p.add_argument("--split", default="train", help="Dataset split (train/validation/test/...)")
    p.add_argument("--streaming", action="store_true", help="Use streaming=True for load_dataset")
    p.add_argument("--revision", default=None, help="Optional dataset revision (hub)")

    p.add_argument("--text-field", default="text", help="Column containing the text")
    p.add_argument("--id-field", default=None, help="Optional id column (fallback: running index)")

    p.add_argument("--output", required=True, help="Output directory; writes one shard per rank")
    p.add_argument("--labels", default="", help="Comma-separated entity labels (e.g. person,company,location)")
    p.add_argument("--labels-json", default="", help="JSON file with list of labels or {label: description}")
    p.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold if supported by API")
    p.add_argument("--include-confidence", action="store_true", help="Include confidence if supported by API")

    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-chars-per-batch", type=int, default=20000)
    p.add_argument("--prefetch-batches", type=int, default=8)

    # GPU perf knobs
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--compile", action="store_true")

    p.add_argument("--flush-every", type=int, default=200)
    return p.parse_args()


def load_labels(args: argparse.Namespace) -> LabelsType:
    if args.labels_json:
        with open(args.labels_json, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            return {str(k): str(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [str(x) for x in obj]
        raise ValueError("--labels-json must be a JSON list or object")

    if not args.labels.strip():
        raise ValueError("Provide --labels (comma-separated) or --labels-json")
    return [x.strip() for x in args.labels.split(",") if x.strip()]


def load_hf_dataset(args: argparse.Namespace):
    """
    Supports:
      - datasets.load_from_disk(path) (Arrow dataset on local disk)
      - datasets.load_dataset(name, config, split=..., streaming=...)
    Returns a Dataset or IterableDataset.
    """
    if os.path.isdir(args.dataset):
        obj = datasets.load_from_disk(args.dataset)
        if isinstance(obj, datasets.DatasetDict):
            if args.split not in obj:
                raise ValueError(f"Split '{args.split}' not found in dataset dict. Available: {list(obj.keys())}")
            return obj[args.split]
        return obj

    # Hub / scripted dataset
    return datasets.load_dataset(
        args.dataset,
        args.config,
        split=args.split,
        streaming=args.streaming,
        revision=args.revision,
    )


def shard_dataset(ds, rank: int, world_size: int, streaming: bool):
    # For map-style datasets, contiguous sharding is cache-friendly.
    if not streaming:
        return ds.shard(num_shards=world_size, index=rank, contiguous=True)
    # IterableDataset also supports shard() in HF datasets.
    return ds.shard(num_shards=world_size, index=rank)


def iter_records_from_dataset(
    ds,
    text_field: str,
    id_field: Optional[str],
) -> Iterable[Record]:
    for idx, ex in enumerate(ds):
        if text_field not in ex:
            raise KeyError(f"Missing text field '{text_field}' in example keys={list(ex.keys())}")
        text = ex[text_field]
        if text is None:
            text = ""
        rid = ex.get(id_field, idx) if id_field else idx
        yield Record(idx=idx, rid=rid, text=str(text), raw=dict(ex))


def batch_records(
    records: Iterable[Record],
    batch_size: int,
    max_chars_per_batch: int,
) -> Iterable[List[Record]]:
    batch: List[Record] = []
    chars = 0
    for r in records:
        rlen = len(r.text)
        if batch and (len(batch) >= batch_size or (chars + rlen) > max_chars_per_batch):
            yield batch
            batch = []
            chars = 0
        batch.append(r)
        chars += rlen
    if batch:
        yield batch


def move_extractor_to_device(extractor: Any, device: torch.device) -> None:
    if hasattr(extractor, "to"):
        try:
            extractor.to(device)
            return
        except Exception:
            pass
    for attr in ("model", "_model", "encoder", "net"):
        m = getattr(extractor, attr, None)
        if m is not None and hasattr(m, "to"):
            try:
                m.to(device)
            except Exception:
                pass


def maybe_compile_extractor(extractor: Any) -> None:
    if not hasattr(torch, "compile"):
        return
    for attr in ("model", "_model", "encoder", "net"):
        m = getattr(extractor, attr, None)
        if isinstance(m, torch.nn.Module):
            try:
                setattr(extractor, attr, torch.compile(m))
            except Exception:
                pass


def build_extract_kwargs(extractor: Any, threshold: float, include_confidence: bool) -> Dict[str, Any]:
    import inspect

    kwargs: Dict[str, Any] = {}
    try:
        sig = inspect.signature(extractor.extract_entities)
        params = sig.parameters
        if "threshold" in params:
            kwargs["threshold"] = threshold
        if "include_confidence" in params:
            kwargs["include_confidence"] = include_confidence
    except Exception:
        pass
    return kwargs


def extract_entities_batch(
    extractor: Any,
    texts: List[str],
    labels: LabelsType,
    kwargs: Dict[str, Any],
) -> List[Dict[str, Any]]:
    try:
        out = extractor.extract_entities(texts, labels, **kwargs)
        if isinstance(out, list):
            return out
        if isinstance(out, dict):
            return [out for _ in texts]
    except Exception:
        pass

    results: List[Dict[str, Any]] = []
    for t in texts:
        results.append(extractor.extract_entities(t, labels, **kwargs))
    return results


def prefetch_batches(batch_iter, q: "queue.Queue[Optional[List[Record]]]") -> None:
    try:
        for b in batch_iter:
            q.put(b)
    finally:
        q.put(None)


def main() -> None:
    args = parse_args()

    state = PartialState()
    rank = state.process_index
    world_size = state.num_processes
    device = state.device

    os.makedirs(args.output, exist_ok=True)
    out_path = os.path.join(args.output, f"part-{rank:05d}.jsonl")

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    labels = load_labels(args)

    # Load + shard dataset per-rank
    ds = load_hf_dataset(args)
    ds = shard_dataset(ds, rank=rank, world_size=world_size, streaming=args.streaming)

    rec_iter = iter_records_from_dataset(ds, args.text_field, args.id_field)
    batches = batch_records(rec_iter, args.batch_size, args.max_chars_per_batch)

    # Model per process / GPU
    extractor = GLiNER2.from_pretrained(args.model)
    move_extractor_to_device(extractor, device)
    if args.compile:
        maybe_compile_extractor(extractor)

    extract_kwargs = build_extract_kwargs(extractor, args.threshold, args.include_confidence)

    use_amp = device.type == "cuda" and (args.fp16 or args.bf16)
    amp_dtype = torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else None)

    q: "queue.Queue[Optional[List[Record]]]" = queue.Queue(maxsize=max(1, args.prefetch_batches))
    t = threading.Thread(target=prefetch_batches, args=(batches, q), daemon=True)
    t.start()

    n = 0
    t0 = time.time()
    last_log = t0

    with open(out_path, "w", encoding="utf-8") as fout, torch.inference_mode():
        while True:
            batch = q.get()
            if batch is None:
                break

            texts = [r.text for r in batch]
            ids = [r.rid for r in batch]

            if use_amp and amp_dtype is not None:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    results = extract_entities_batch(extractor, texts, labels, extract_kwargs)
            else:
                results = extract_entities_batch(extractor, texts, labels, extract_kwargs)

            for rid, text, res in zip(ids, texts, results):
                line = {
                    "id": rid,
                    "text": text,
                    **(res if isinstance(res, dict) else {"result": res}),
                }
                fout.write(json.dumps(line, ensure_ascii=False) + "\n")
                n += 1

            if (n % args.flush_every) == 0:
                fout.flush()

            now = time.time()
            if now - last_log >= 5.0:
                dt = now - t0
                rps = n / max(dt, 1e-9)
                print(f"[rank {rank}/{world_size}] processed={n}  rps={rps:.2f}  device={device}", flush=True)
                last_log = now

        fout.flush()

    dt = time.time() - t0
    rps = n / max(dt, 1e-9)
    print(f"[rank {rank}/{world_size}] DONE processed={n} in {dt:.2f}s  rps={rps:.2f}  wrote={out_path}", flush=True)


if __name__ == "__main__":
    main()

"""
pip install gliner2 accelerate datasets torch
accelerate launch --num_processes 4 multi_gpu_gliner2_hf_dataset_infer.py \
  --dataset <your_dataset_or_path> --split train \
  --text-field text \
  --labels person,company,location \
  --output out --batch-size 64 --bf16 --streaming
"""
