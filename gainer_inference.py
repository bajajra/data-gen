#!/usr/bin/env python3
"""
multi_gpu_gliner2_infer.py

High-throughput multi-GPU entity extraction with GLiNER2 using data-parallel sharding.
Each process loads its own GLiNER2 model on its own device (GPU) and processes a shard
of the input.

Input:
  - .jsonl: each line is a JSON object containing `text_field` (default: "text")
  - .txt: each line is treated as raw text

Output:
  - Writes one shard per rank: <output_dir>/part-00000.jsonl, part-00001.jsonl, ...
  - Each output line is JSON with at minimum: {"id": ..., "text": ..., "entities": ...}

Run (recommended):
  pip install gliner2 accelerate torch
  accelerate launch --num_processes 4 multi_gpu_gliner2_infer.py \
      --input data.jsonl --output out \
      --labels company,person,product,location --batch-size 64 --fp16

Notes:
  - The script tries to call extract_entities() in batch mode (list[str]) if supported;
    otherwise it falls back to per-text calls.
  - It attempts to pass `threshold` / `include_confidence` only if the installed
    gliner2 version supports those parameters.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
from accelerate import PartialState
from gliner2 import GLiNER2


LabelsType = Union[List[str], Dict[str, str]]  # list of labels OR {label: description}


@dataclass
class Record:
    idx: int
    rid: Union[int, str]
    text: str
    raw: Optional[dict] = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="fastino/gliner2-base-v1", help="HF model id for GLiNER2")
    p.add_argument("--input", required=True, help="Path to .jsonl or .txt input")
    p.add_argument("--output", required=True, help="Output directory (will write one shard per rank)")
    p.add_argument("--text-field", default="text", help="JSONL field containing the text")
    p.add_argument("--id-field", default="id", help="JSONL field containing the id (fallback: line index)")
    p.add_argument("--labels", default="", help="Comma-separated entity labels (e.g. person,company,location)")
    p.add_argument(
        "--labels-json",
        default="",
        help="Path to JSON file containing either a list of labels OR a {label: description} mapping",
    )
    p.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold if supported by API")
    p.add_argument("--include-confidence", action="store_true", help="Include confidence if supported by API")
    p.add_argument("--batch-size", type=int, default=64, help="Max records per forward")
    p.add_argument(
        "--max-chars-per-batch",
        type=int,
        default=20000,
        help="Soft cap to avoid pathological long-text batches (sum of char lengths).",
    )
    p.add_argument(
        "--prefetch-batches",
        type=int,
        default=8,
        help="How many batches to prefetch in a background thread (per rank).",
    )

    # GPU perf knobs
    p.add_argument("--fp16", action="store_true", help="Use fp16 autocast on CUDA")
    p.add_argument("--bf16", action="store_true", help="Use bf16 autocast on CUDA (Ampere+)")
    p.add_argument("--compile", action="store_true", help="Try torch.compile() where possible")

    p.add_argument("--flush-every", type=int, default=200, help="Flush output every N records")
    return p.parse_args()


def load_labels(args: argparse.Namespace) -> LabelsType:
    if args.labels_json:
        with open(args.labels_json, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            # {label: description}
            return {str(k): str(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [str(x) for x in obj]
        raise ValueError("--labels-json must be a JSON list or object")

    if not args.labels.strip():
        raise ValueError("Provide --labels (comma-separated) or --labels-json")
    return [x.strip() for x in args.labels.split(",") if x.strip()]


def iter_records(
    path: str,
    text_field: str,
    id_field: str,
    rank: int,
    world_size: int,
) -> Iterable[Record]:
    """
    Streaming reader that shards by line index:
      line_idx % world_size == rank
    This avoids loading the whole dataset and keeps ordering deterministic per-shard.
    """
    is_jsonl = path.endswith(".jsonl") or path.endswith(".jsonl.gz")  # gzip not handled here
    if path.endswith(".gz"):
        raise ValueError("This script does not handle .gz directly. Decompress first or extend reader.")

    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if (idx % world_size) != rank:
                continue
            line = line.strip()
            if not line:
                continue

            if is_jsonl:
                obj = json.loads(line)
                text = obj.get(text_field, "")
                if text is None:
                    text = ""
                rid = obj.get(id_field, idx)
                yield Record(idx=idx, rid=rid, text=str(text), raw=obj)
            else:
                yield Record(idx=idx, rid=idx, text=line, raw=None)


def batch_records(
    records: Iterable[Record],
    batch_size: int,
    max_chars_per_batch: int,
) -> Iterable[List[Record]]:
    batch: List[Record] = []
    chars = 0
    for r in records:
        # crude length heuristic to keep latency sane
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
    """
    Best-effort device placement across potential GLiNER2 internal structures.
    """
    # If the wrapper itself supports `.to(device)`
    if hasattr(extractor, "to"):
        try:
            extractor.to(device)
            return
        except Exception:
            pass

    # Try common internal attributes
    for attr in ("model", "_model", "encoder", "net"):
        m = getattr(extractor, attr, None)
        if m is not None and hasattr(m, "to"):
            try:
                m.to(device)
            except Exception:
                pass


def maybe_compile_extractor(extractor: Any) -> None:
    """
    Best-effort torch.compile. If GLiNER2 exposes an underlying torch.nn.Module, try compiling it.
    """
    if not hasattr(torch, "compile"):
        return
    for attr in ("model", "_model", "encoder", "net"):
        m = getattr(extractor, attr, None)
        if isinstance(m, torch.nn.Module):
            try:
                compiled = torch.compile(m)  # type: ignore[attr-defined]
                setattr(extractor, attr, compiled)
            except Exception:
                pass


def build_extract_kwargs(extractor: Any, threshold: float, include_confidence: bool) -> Dict[str, Any]:
    """
    Only pass kwargs supported by the installed gliner2 version.
    """
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
        # If introspection fails, be conservative: don't pass anything extra
        pass
    return kwargs


def extract_entities_batch(
    extractor: Any,
    texts: List[str],
    labels: LabelsType,
    kwargs: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Try batch-mode first; fallback to per-text if the installed API doesn't support list inputs.
    Returns a list aligned with `texts`.
    """
    # Attempt vectorized call (if supported by library)
    try:
        out = extractor.extract_entities(texts, labels, **kwargs)
        # Possible shapes:
        # - list[dict] (ideal)
        # - dict (if library ignored list and treated as one input)
        if isinstance(out, list):
            return out
        if isinstance(out, dict):
            # treat as single result
            return [out for _ in texts]
    except TypeError:
        pass
    except Exception:
        # If batch call fails for any reason, fallback
        pass

    # Fallback: per-text calls
    results: List[Dict[str, Any]] = []
    for t in texts:
        results.append(extractor.extract_entities(t, labels, **kwargs))
    return results


def prefetch_batches(
    batch_iter: Iterable[List[Record]],
    q: "queue.Queue[Optional[List[Record]]]",
) -> None:
    try:
        for b in batch_iter:
            q.put(b)
    finally:
        q.put(None)  # sentinel


def main() -> None:
    args = parse_args()

    state = PartialState()
    rank = state.process_index
    world_size = state.num_processes
    device = state.device

    os.makedirs(args.output, exist_ok=True)
    out_path = os.path.join(args.output, f"part-{rank:05d}.jsonl")

    # Perf knobs (safe to set even if CPU)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    labels = load_labels(args)

    # Load model (one per process)
    extractor = GLiNER2.from_pretrained(args.model)

    # Try to move to GPU if available (best-effort).
    move_extractor_to_device(extractor, device)

    # Optional compile
    if args.compile:
        maybe_compile_extractor(extractor)

    extract_kwargs = build_extract_kwargs(extractor, args.threshold, args.include_confidence)

    # autocast setup
    use_amp = device.type == "cuda" and (args.fp16 or args.bf16)
    amp_dtype = torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else None)

    # Stream + shard input
    rec_iter = iter_records(args.input, args.text_field, args.id_field, rank, world_size)
    batches = batch_records(rec_iter, args.batch_size, args.max_chars_per_batch)

    # Background prefetcher (keeps GPU busier)
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

            # Write JSONL
            for rid, text, res in zip(ids, texts, results):
                # Normalize output a bit: keep the usual {'entities': {...}} under `res`
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
