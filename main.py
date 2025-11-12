import requests
from prompts.translate_prompt import prompt
from typing import Callable, Optional, Dict, Any
from datasets import load_dataset, concatenate_datasets, load_from_disk
from pathlib import Path


def get_llm_response(text: str) -> str:
    headers = {
    'Content-Type': 'application/json',

}

    json_data = {
        'model': 'Qwen/Qwen3-4B-Instruct-2507-FP8',
        'messages': [
            {
                'role': 'system',
                'content': prompt,
            },
            {
                'role': 'user',
                'content': text,
            },
        ],
    }

    response = requests.post('http://localhost:8082/v1/chat/completions', headers=headers, json=json_data)
    return response.json()['choices'][0]['message']['content']

def translate_text(source_text: str, target_language: str) -> str:
    request_template = """**Target language:** <target_language>

**Source text**:
<source_text>
"""
    filled_prompt = request_template.replace("<target_language>", target_language).replace("<source_text>", source_text)
    translation = get_llm_response(filled_prompt)
    return translation

def choose_target_language():
    """Function to choose target language with equal probability."""
    import random
    languages = ['Russian', 'French', 'Spanish', 'German', 'Italian', 'Portuguese', 'Dutch', "Ukrainian", "Polish"]
    return random.choice(languages)

def checkpointed_map(
    ds,                               # a datasets.Dataset (single split)
    map_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
    out_dir: str,
    *,
    num_shards: int = 64,
    num_proc: int = 8,
    batched: bool = True,
    batch_size: Optional[int] = 1000,
    writer_batch_size: Optional[int] = 1000,
    map_desc_prefix: str = "map",
    save_final: bool = True,
):
    """
    Process `ds` in contiguous shards; each processed shard is saved to `out_dir/shard_{i}`.
    If interrupted, re-running will resume from the first missing shard. Returns the final
    concatenated dataset (loading shards from disk if present).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    shard_dirs = [out / f"shard_{i:04d}" for i in range(num_shards)]

    # Process any missing shards
    for i, shard_path in enumerate(shard_dirs):
        if shard_path.exists():
            print(f"[skip] shard {i}/{num_shards-1}: already at {shard_path}")
            continue

        # Use contiguous shards so concatenation restores original order
        shard = ds.shard(num_shards=num_shards, index=i, contiguous=True)
        print(f"[run ] shard {i}/{num_shards-1}: size ~{len(shard)} -> {shard_path}")

        processed = shard.map(
            map_fn,
            batched=batched,
            batch_size=batch_size,
            num_proc=num_proc,
            writer_batch_size=writer_batch_size,
            load_from_cache_file=True,   # reuse internal cache per worker if available
            desc=f"{map_desc_prefix} [shard {i}/{num_shards-1}]",
        )

        processed.save_to_disk(str(shard_path))

    # Load all shards in order (contiguous=True keeps the original ordering across shards)
    pieces = [load_from_disk(str(p)) for p in shard_dirs]
    final_ds = concatenate_datasets(pieces)

    if save_final:
        final_path = out / "final"
        final_ds.save_to_disk(str(final_path))
        print(f"[done] final dataset saved to {final_path}")

    return final_ds

def preprocess(batch):
    results = []
    languages = []
    for text in batch["text"]:
        target_language = choose_target_language()
        
        try:
            translated_text = translate_text(text, target_language)
        except Exception as e:
            print(f"Error translating text: {e}")
            translated_text = text  # Fallback to original text on error
            target_language = "English"

        languages.append(target_language)
        results.append(translated_text)
    batch["translated_text"] = results
    batch["target_language"] = languages
    return batch

if __name__ == "__main__":
    ds = load_dataset("thebajajra/Ecomniverse-sampled") 
    final = checkpointed_map(
            ds['train'],
            preprocess,
            out_dir="ecomniverse-translated-euro",
            num_shards=32000,         # tune for your dataset size / failure domains
            num_proc=128,            # your request: use 8 processes in map()
            batched=True,
            batch_size=1000,
            writer_batch_size=1000,
            save_final=True,
        )

        # Use the result
    print(final)