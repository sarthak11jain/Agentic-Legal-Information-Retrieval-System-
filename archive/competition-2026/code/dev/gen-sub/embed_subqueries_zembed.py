from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Iterable

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[3]
UTILS_DIR = BASE_DIR / "code/dev/utils"
sys.path.append(str(UTILS_DIR))

from call_embedder import call_embedder_batch  # noqa: E402


DEFAULT_INPUT_PATH = BASE_DIR / "data/test_subqueries_qwen27b.parquet"
DEFAULT_OUTPUT_PATH = BASE_DIR / "data/test_subqueries_qwen27b_zembed.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed generated German subqueries with ZeroEntropy zembed-1."
    )
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--text-column", default="sub_query_de")
    parser.add_argument("--embedding-column", default="zembed_embedding")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--model", default="zembed-1")
    parser.add_argument("--dimensions", type=int, default=2560)
    parser.add_argument("--input-type", choices=["query", "document"], default="query")
    parser.add_argument("--encoding-format", choices=["float", "base64"], default="float")
    parser.add_argument("--latency", choices=["fast", "slow"], default="slow")
    parser.add_argument("--timeout-s", type=int, default=180)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--retry-sleep-s", type=float, default=2.0)
    parser.add_argument("--row-start", type=int, default=1, help="1-indexed inclusive row start.")
    parser.add_argument("--row-end", type=int, default=0, help="1-indexed inclusive row end; 0 means all rows.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def batched(values: list[int], batch_size: int) -> Iterable[list[int]]:
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def load_or_initialize(args: argparse.Namespace) -> pd.DataFrame:
    source = pd.read_parquet(args.input_path).reset_index(drop=True)
    if args.text_column not in source.columns:
        raise SystemExit(f"{args.input_path} does not contain text column {args.text_column!r}")

    if args.output_path.exists() and not args.overwrite:
        existing = pd.read_parquet(args.output_path).reset_index(drop=True)
        if len(existing) != len(source):
            raise SystemExit(
                f"Existing output has {len(existing)} rows but input has {len(source)} rows. "
                "Use --overwrite to regenerate."
            )
        for column in source.columns:
            if column not in existing.columns:
                existing[column] = source[column]
        return existing

    output = source.copy()
    output[args.embedding_column] = None
    return output


def needs_embedding(value: object) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except ValueError:
        return False


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")

    load_dotenv(BASE_DIR / ".env")
    output = load_or_initialize(args)

    row_start = max(args.row_start, 1)
    row_end = args.row_end if args.row_end > 0 else len(output)
    row_end = min(row_end, len(output))
    if row_start > row_end:
        raise SystemExit(f"Invalid row range: {row_start}-{row_end}")

    row_indices = list(range(row_start - 1, row_end))
    pending_indices = [
        idx
        for idx in row_indices
        if needs_embedding(output.at[idx, args.embedding_column])
    ]

    print(f"Input          : {args.input_path}")
    print(f"Output         : {args.output_path}")
    print(f"Rows           : {len(output):,}")
    print(f"Row range      : {row_start}-{row_end}")
    print(f"Pending embeds : {len(pending_indices):,}")
    print(f"Batch size     : {args.batch_size}")
    print(f"Model          : {args.model}")
    print(f"Input type     : {args.input_type}")
    print(f"Latency        : {args.latency}")

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = 0
    total_batches = (len(pending_indices) + args.batch_size - 1) // args.batch_size

    for batch_no, indices in enumerate(batched(pending_indices, args.batch_size), start=1):
        texts = [str(output.at[idx, args.text_column]) for idx in indices]
        last_error: Exception | None = None
        embeddings: list[list[float]] | None = None
        for attempt in range(1, args.max_retries + 1):
            try:
                embeddings = call_embedder_batch(
                    texts,
                    model=args.model,
                    dimensions=args.dimensions,
                    input_type=args.input_type,
                    encoding_format=args.encoding_format,
                    latency=args.latency,
                    timeout_s=args.timeout_s,
                )
                break
            except Exception as error:  # noqa: BLE001 - keep batch retry robust for API jobs.
                last_error = error
                if attempt == args.max_retries:
                    break
                sleep_s = args.retry_sleep_s * attempt
                print(
                    f"Batch {batch_no}/{total_batches} failed on attempt {attempt}: "
                    f"{error}. Retrying in {sleep_s:.1f}s..."
                )
                time.sleep(sleep_s)

        if embeddings is None:
            raise RuntimeError(f"Batch {batch_no}/{total_batches} failed: {last_error}")

        for idx, embedding in zip(indices, embeddings, strict=True):
            output.at[idx, args.embedding_column] = embedding

        completed += len(indices)
        output.to_parquet(args.output_path, index=False)
        print(
            f"Completed batch {batch_no}/{total_batches}: "
            f"{completed:,}/{len(pending_indices):,} embedded"
        )

    output.to_parquet(args.output_path, index=False)
    print(f"Saved          : {args.output_path}")


if __name__ == "__main__":
    main()
