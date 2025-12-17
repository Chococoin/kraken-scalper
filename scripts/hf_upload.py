#!/usr/bin/env python3
"""
Upload data files to HuggingFace.

Usage:
    python scripts/hf_upload.py --repo-id USER/DATASET --data-dir data --token TOKEN

This script uses huggingface_hub which handles LFS automatically.
"""

import argparse
import json
import os
import sys
from pathlib import Path

try:
    from huggingface_hub import HfApi
except ImportError:
    print(json.dumps({"success": False, "error": "huggingface_hub not installed"}))
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Upload data to HuggingFace")
    parser.add_argument("--repo-id", required=True, help="Repository ID (user/dataset)")
    parser.add_argument("--data-dir", required=True, help="Data directory to upload")
    parser.add_argument("--token", required=True, help="HuggingFace token")
    parser.add_argument("--json", action="store_true", help="Output JSON (for Rust integration)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    if not data_dir.exists():
        result = {"success": False, "error": f"Data directory '{data_dir}' not found"}
        if args.json:
            print(json.dumps(result))
        else:
            print(f"Error: {result['error']}")
        sys.exit(1)

    # Count files
    parquet_files = list(data_dir.rglob("*.parquet"))
    if not parquet_files:
        result = {"success": True, "files_uploaded": 0, "bytes_uploaded": 0}
        if args.json:
            print(json.dumps(result))
        else:
            print("No files to upload")
        sys.exit(0)

    total_size = sum(f.stat().st_size for f in parquet_files)

    if not args.json:
        print(f"Repository: {args.repo_id}")
        print(f"Files to upload: {len(parquet_files)}")
        print(f"Total size: {total_size / 1_000_000:.2f} MB")
        print()
        print("Uploading to HuggingFace...")

    api = HfApi()

    try:
        api.upload_folder(
            folder_path=str(data_dir),
            repo_id=args.repo_id,
            repo_type="dataset",
            path_in_repo="data",
            token=args.token,
            commit_message=f"Upload {len(parquet_files)} data files ({total_size / 1_000_000:.2f} MB)",
        )

        result = {
            "success": True,
            "files_uploaded": len(parquet_files),
            "bytes_uploaded": total_size,
        }

        if args.json:
            print(json.dumps(result))
        else:
            print()
            print("=" * 60)
            print("Upload successful!")
            print("=" * 60)
            print(f"\nDataset URL: https://huggingface.co/datasets/{args.repo_id}")
            print(f"Files uploaded: {len(parquet_files)}")

    except Exception as e:
        result = {"success": False, "error": str(e)}
        if args.json:
            print(json.dumps(result))
        else:
            print(f"Upload failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
