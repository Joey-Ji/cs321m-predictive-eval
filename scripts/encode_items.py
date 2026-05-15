"""Stage 2a (part 1): encode every unique training item with a sentence transformer.

The encoder consumes only raw `item_content`. Benchmark and condition are
recorded as one-hot side features (built from the full training set) and
concatenated to the embedding at head input — see `src/features.py`.

Reads data/joined.parquet (or data/items.parquet directly).
Writes (into --out, default data/embeddings/mpnet_v1/):
  item_embeddings.npy        — float32 [n_items, d]
  item_id_order.json         — list[str] indexing rows of the npy
  encoder_meta.json          — encoder name, dim, count, representation
  item_side_features.npy     — float32 [n_items, side_feature_dim]
  side_feature_meta.json     — one-hot vocab + dims for benchmark/condition

Default encoder: sentence-transformers/all-mpnet-base-v2 (768-d).

Usage:
    python scripts/encode_items.py
    python scripts/encode_items.py --encoder BAAI/bge-large-en-v1.5 --batch 64
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features import (
    EMBEDDING_REPRESENTATION_VERSION,
    RAW_ITEM_TEXT_VERSION,
    build_side_feature_vocab,
    encode_side_features,
)


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
    dummy_dim: int,
) -> None:
    import numpy as np
    import pyarrow.parquet as pq

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading {joined_path} ...")
    table = pq.read_table(joined_path, columns=["item_id", "benchmark", "condition", "item_content"])
    rows = table.to_pylist()

    vocab = build_side_feature_vocab(rows)
    print(
        f"  side features: benchmark_dim={vocab['benchmark_dim']} "
        f"condition_dim={vocab['condition_dim']} "
        f"side_feature_dim={vocab['side_feature_dim']}"
    )

    seen_texts: dict[str, str] = {}
    seen_rows: dict[str, dict] = {}
    saw_null_item_id = False
    for row in rows:
        raw_iid = row.get("item_id")
        saw_null_item_id = saw_null_item_id or raw_iid is None
        iid = "" if raw_iid is None else str(raw_iid)
        if iid not in seen_texts:
            seen_texts[iid] = ("" if row.get("item_content") is None else str(row.get("item_content")))[:max_chars]
            seen_rows[iid] = row
    print(f"  unique items: {len(seen_texts):,}")
    if saw_null_item_id:
        print("WARN: one or more rows had null item_id; deduplicated under empty-string item id", file=sys.stderr)

    item_id_order = list(seen_texts.keys())
    texts = [seen_texts[iid] for iid in item_id_order]

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

    side_features = np.stack(
        [encode_side_features(seen_rows[iid], vocab) for iid in item_id_order]
    ).astype(np.float32)

    np.save(out_dir / "item_embeddings.npy", emb)
    np.save(out_dir / "item_side_features.npy", side_features)
    (out_dir / "item_id_order.json").write_text(json.dumps(item_id_order))
    (out_dir / "encoder_meta.json").write_text(
        json.dumps(
            {
                "encoder": encoder,
                "dim": dim,
                "count": len(item_id_order),
                "max_chars": max_chars,
                "representation_version": EMBEDDING_REPRESENTATION_VERSION,
                "feature_text_version": RAW_ITEM_TEXT_VERSION,
                "dummy": is_dummy,
                "batch": batch,
                "command": " ".join(sys.argv),
            },
            indent=2,
        )
    )
    (out_dir / "side_feature_meta.json").write_text(json.dumps(vocab, indent=2))
    print(f"Done. Wrote {emb.shape} to {out_dir}/item_embeddings.npy")
    print(f"      Wrote {side_features.shape} to {out_dir}/item_side_features.npy")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--joined", default="data/joined.parquet", type=Path)
    parser.add_argument("--out", default="data/embeddings/mpnet_v1", type=Path)
    parser.add_argument("--encoder", default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--max-chars", type=int, default=4000, help="Truncate item text to this many chars before encoding.")
    parser.add_argument("--dummy-dim", type=int, default=8, help="Embedding dimension when --encoder dummy is used.")
    args = parser.parse_args()
    main(args.joined, args.out, args.encoder, args.batch, args.max_chars, args.dummy_dim)
