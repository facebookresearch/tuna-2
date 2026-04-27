"""Download the CountBenchQA dataset for offline evaluation.

Usage (on a machine with internet access):
    python download_countbench.py [--save_dir /path/to/cache]

The dataset will be cached to the HuggingFace default cache directory
(~/.cache/huggingface/datasets) unless --save_dir is specified.
After downloading, copy the cache to the devvm and evaluations will
load from the local cache automatically.
"""

from datasets import load_dataset

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Optional directory to save the dataset. "
        "If not set, uses default HF cache (~/.cache/huggingface/datasets).",
    )
    args = parser.parse_args()

    print("Downloading vikhyatk/CountBenchQA ...")
    ds = load_dataset("vikhyatk/CountBenchQA", cache_dir=args.save_dir)
    print(f"Done. Splits: {list(ds.keys())}, total samples: {sum(len(s) for s in ds.values())}")
    if args.save_dir:
        print(f"Saved to: {args.save_dir}")
    else:
        print("Saved to default HF cache: ~/.cache/huggingface/datasets")
