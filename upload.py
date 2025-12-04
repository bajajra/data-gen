#!/usr/bin/env python3
"""
upload.py

Upload each subfolder of a local ROOT_DIR to a single Hugging Face dataset repo,
appending new data and skipping anything already uploaded.

- Each immediate subfolder under ROOT_DIR is treated as one "chunk" of data.
- On the Hub, each chunk lands under: data/<subfolder_name> within the same dataset repo.
- A local JSON state file tracks what was uploaded (and a quick signature).
- Reruns only upload new or changed subfolders.

Requirements:
  pip install -U huggingface-hub

Auth:
  Either set env var HUGGINGFACE_TOKEN=hf_... or pass --token.
  (You can also have `huggingface-cli login` done already.)

Example:
python upload.py --root ecomniverse-translated-euro --repo-id thebajajra/Ecomniverse-multilingual
"""

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any

from huggingface_hub import HfApi, login


IGNORES = {
    ".DS_Store",
    "Thumbs.db",
    "__pycache__",
    ".ipynb_checkpoints",
}


@dataclass
class Args:
    root: Path
    repo_id: str
    path_in_repo: str
    state_file: Path
    private: bool
    token: str | None
    dry_run: bool


def parse_args() -> Args:
    p = argparse.ArgumentParser(description="Append multiple local dataset folders into a single HF dataset repo.")
    p.add_argument("--root", required=True, type=Path, help="Root folder containing many dataset subfolders.")
    p.add_argument("--repo-id", required=True, help="Target dataset repo on HF Hub, e.g. 'username/my-dataset'.")
    p.add_argument("--path-in-repo", default="data", help="Base folder inside the repo to place uploads. Default: data")
    p.add_argument("--state-file", type=Path, default=Path(".hf_upload_state.json"),
                   help="Path to local JSON ledger. Default: .hf_upload_state.json")
    p.add_argument("--private", action="store_true", help="Create/use a private dataset repo.")
    p.add_argument("--token", default=os.environ.get("HUGGINGFACE_TOKEN"), help="HF token (or set HUGGINGFACE_TOKEN).")
    p.add_argument("--dry-run", action="store_true", help="Show what would be done without uploading.")
    a = p.parse_args()

    return Args(
        root=a.root.resolve(),
        repo_id=a.repo_id,
        path_in_repo=a.path_in_repo.strip("/"),
        state_file=a.state_file.resolve(),
        private=bool(a.private),
        token=a.token,
        dry_run=bool(a.dry_run),
    )


def load_state(path: Path) -> Dict[str, Any]:
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {"uploaded": {}, "repo_id": None, "path_in_repo": None}


def save_state(path: Path, state: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)
    tmp.replace(path)


def is_ignored(name: str) -> bool:
    lower = name.lower()
    return name in IGNORES or any(seg in IGNORES for seg in name.split(os.sep)) or lower.endswith(".tmp")


def folder_signature(folder: Path) -> str:
    """
    Fast, deterministic digest of a folder based on relative paths, mtimes, and sizes.
    (Good enough to avoid re-uploads; not a full content hash to stay quick.)
    """
    h = hashlib.sha256()
    for p in sorted(folder.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(folder).as_posix()
        if is_ignored(rel):
            continue
        st = p.stat()
        h.update(rel.encode("utf-8"))
        h.update(str(int(st.st_mtime)).encode("utf-8"))
        h.update(str(st.st_size).encode("utf-8"))
    return h.hexdigest()


def list_immediate_subfolders(root: Path) -> list[Path]:
    return [p for p in sorted(root.iterdir()) if p.is_dir() and not is_ignored(p.name)]


def ensure_repo(api: HfApi, repo_id: str, private: bool) -> None:
    # Create the dataset repo if it doesn't exist; no-op if it does.
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)


def upload_one_folder(api: HfApi, src: Path, repo_id: str, path_in_repo_base: str, dry_run: bool) -> dict:
    dest = f"{path_in_repo_base}/{src.name}"
    msg = f"Add/update dataset chunk: {src.name} -> {dest}"
    if dry_run:
        print(f"[DRY RUN] Would upload: {src}  ->  hf://{repo_id}/{dest}")
        return {"oid": None, "commit_url": None}

    # Upload local folder into the dataset repo under the 'dest' directory.
    # If the files/paths already exist, they will be overwritten in this commit.
    # (Behavior per upload_folder & CommitOperationAdd docs.)
    commit = api.upload_folder(
        folder_path=str(src),
        repo_id=repo_id,
        repo_type="dataset",
        path_in_repo=dest,
        commit_message=msg,
        ignore_patterns=["**/*.json", "*.json"]
        # You can also use allow_patterns / ignore_patterns to filter, if needed.
    )
    # `upload_folder` returns a CommitInfo (commit hash/url, etc.)
    # See docs: CommitInfo has fields like `oid` (hash) and `commit_url`.
    return {"oid": getattr(commit, "oid", None), "commit_url": getattr(commit, "commit_url", None)}


def main():
    args = parse_args()

    if not args.root.is_dir():
        print(f"ERROR: --root '{args.root}' is not a directory.", file=sys.stderr)
        sys.exit(2)

    # Login (no-op if you already ran `huggingface-cli login` on this machine).
    if args.token:
        login(token=args.token)

    api = HfApi()

    # Ensure dataset repo exists.
    ensure_repo(api, args.repo_id, args.private)

    state = load_state(args.state_file)
    state.setdefault("uploaded", {})
    state["repo_id"] = args.repo_id
    state["path_in_repo"] = args.path_in_repo

    subfolders = list_immediate_subfolders(args.root)
    if not subfolders:
        print(f"No subfolders found under {args.root}. Nothing to do.")
        return

    print(f"Found {len(subfolders)} subfolders under {args.root}")
    uploaded_count = 0
    skipped_count = 0

    for folder in subfolders:
        sig = folder_signature(folder)
        rec = state["uploaded"].get(folder.name)

        if rec and rec.get("signature") == sig:
            print(f"Skip (unchanged/already uploaded): {folder.name}")
            skipped_count += 1
            continue

        print(f"Uploading: {folder.name}")
        try:
            info = upload_one_folder(api, folder, args.repo_id, args.path_in_repo, args.dry_run)
            uploaded_count += 1
            state["uploaded"][folder.name] = {
                "signature": sig,
                "uploaded_at": datetime.now(timezone.utc).isoformat(),
                "commit_oid": info.get("oid"),
                "commit_url": info.get("commit_url"),
                "path_in_repo": f"{args.path_in_repo}/{folder.name}",
            }
            save_state(args.state_file, state)
        except Exception as e:
            print(f"ERROR uploading {folder.name}: {e}", file=sys.stderr)
            # Record the failure without updating signature, so it retries next run
            state["uploaded"].setdefault(folder.name, {})
            state["uploaded"][folder.name]["last_error"] = str(e)
            save_state(args.state_file, state)

    print(f"Done. Uploaded: {uploaded_count}, skipped: {skipped_count}.")
    print(f"Ledger: {args.state_file}")


if __name__ == "__main__":
    main()