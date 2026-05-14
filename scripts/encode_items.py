"""Stage 2a (part 1): encode every unique training item with a sentence transformer.

Reads data/joined.parquet (or data/items.parquet directly).
Writes:
  data/embeddings/item_embeddings.npy   — float32 [n_items, d]
  data/embeddings/item_id_order.json    — list[str] indexing rows of the npy
  data/embeddings/encoder_meta.json     — encoder name, dim, count, text version

Default encoder: sentence-transformers/all-mpnet-base-v2 (768-d, fast).
Upgrade options: BAAI/bge-large-en-v1.5 (1024-d, MTEB 64.23).

Usage:
    python scripts/encode_items.py
    python scripts/encode_items.py --encoder BAAI/bge-large-en-v1.5 --batch 64
    python scripts/encode_items.py --feature-text-version benchmark_condition_item_v1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features import FEATURE_TEXT_VERSION, RAW_ITEM_TEXT_VERSION, build_item_feature_text


TEXT_VERSIONS = {RAW_ITEM_TEXT_VERSION, FEATURE_TEXT_VERSION}


def _dummy_embedding(text: str, dim: int):
    import numpy as np

    digest = hashlib.sha256(f"dummy-encoder-v1|{text}".encode()).digest()
    seed = int.from_bytes(digest[:8], "big", signed=False)
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 1.0, size=dim).astype(np.float32)


def main(
    joined_path: Path,
    out_dir: Path,
    encoder: str,
    batch: int,
    max_chars: int,
    feature_text_version: str,
    dummy_dim: int,
) -> None:
    import numpy as np
    import pyarrow.parquet as pq

    if feature_text_version not in TEXT_VERSIONS:
        raise ValueError(f"unsupported feature text version {feature_text_version!r}; expected one of {sorted(TEXT_VERSIONS)}")

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading {joined_path} ...")
    columns = ["item_id", "item_content"]
    if feature_text_version == FEATURE_TEXT_VERSION:
        columns = ["item_id", "benchmark", "condition", "item_content"]
    table = pq.read_table(joined_path, columns=columns)
    rows = table.to_pylist()

    seen: dict[str, str] = {}
    saw_null_item_id = False
    for row in rows:
        raw_iid = row.get("item_id")
        saw_null_item_id = saw_null_item_id or raw_iid is None
        iid = "" if raw_iid is None else str(raw_iid)
        if iid not in seen:
            if feature_text_version == RAW_ITEM_TEXT_VERSION:
                seen[iid] = ("" if row.get("item_content") is None else str(row.get("item_content")))[:max_chars]
            else:
                seen[iid] = build_item_feature_text(row, max_chars=max_chars)
    print(f"  unique items: {len(seen):,}")
    if saw_null_item_id:
        print("WARN: one or more rows had null item_id; deduplicated under empty-string item id", file=sys.stderr)

    item_id_order = list(seen.keys())
    texts = [seen[iid] for iid in item_id_order]

    is_dummy = encoder == "dummy"
    if is_dummy:
        dim = dummy_dim
        print(f"Encoding {len(texts):,} items with deterministic dummy encoder dim={dim} ...")
        emb = np.stack([_dummy_embedding(text, dim) for text in texts]).astype(np.float32)
    else:
        from sentence_transformers import SentenceTransformer

        print(f"Loading encoder {encoder} ...")
        model = SentenceTransformer(encoder)
        dim = int(model.get_sentence_embedding_dimension())
        print(f"  dim: {dim}")

        print(f"Encoding {len(texts):,} items at batch={batch} ...")
        emb = model.encode(
            texts,
            batch_size=batch,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=False,
        ).astype(np.float32)

    np.save(out_dir / "item_embeddings.npy", emb)
    (out_dir / "item_id_order.json").write_text(json.dumps(item_id_order))
    (out_dir / "encoder_meta.json").write_text(
        json.dumps(
            {
                "encoder": encoder,
                "dim": dim,
                "count": len(item_id_order),
                "max_chars": max_chars,
                "feature_text_version": feature_text_version,
                "dummy": is_dummy,
                "batch": batch,
                "command": " ".join(sys.argv),
            },
            indent=2,
        )
    )
    print(f"Done. Wrote {emb.shape} to {out_dir}/item_embeddings.npy")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--joined", default="data/joined.parquet", type=Path)
    parser.add_argument("--out", default="data/embeddings", type=Path)
    parser.add_argument("--encoder", default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--max-chars", type=int, default=4000, help="Truncate item text to this many chars before encoding.")
    parser.add_argument(
        "--feature-text-version",
        default=RAW_ITEM_TEXT_VERSION,
        choices=sorted(TEXT_VERSIONS),
        help="Text format to encode. Defaults to legacy raw item_content for v1_irt compatibility.",
    )
    parser.add_argument("--dummy-dim", type=int, default=8, help="Embedding dimension when --encoder dummy is used.")
    args = parser.parse_args()
    main(args.joined, args.out, args.encoder, args.batch, args.max_chars, args.feature_text_version, args.dummy_dim)
