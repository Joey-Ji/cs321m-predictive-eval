"""Stage 2a (part 1): encode every unique training item with a sentence transformer.

Reads data/joined.parquet (or data/items.parquet directly).
Writes:
  data/embeddings/item_embeddings.npy   — float32 [n_items, d]
  data/embeddings/item_id_order.json    — list[str] indexing rows of the npy
  data/embeddings/encoder_meta.json     — encoder name, dim, count, feature text version

Default encoder: sentence-transformers/all-mpnet-base-v2 (768-d, fast).
Upgrade options: BAAI/bge-large-en-v1.5 (1024-d, MTEB 64.23).

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

from src.features import FEATURE_TEXT_VERSION, build_item_feature_text


def _dummy_embedding(item_id: str, dim: int):
    import numpy as np

    digest = hashlib.sha256(f"dummy-encoder-v1|{item_id}".encode()).digest()
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

    if feature_text_version != FEATURE_TEXT_VERSION:
        raise ValueError(f"unsupported feature text version {feature_text_version!r}; expected {FEATURE_TEXT_VERSION!r}")

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading {joined_path} ...")
    table = pq.read_table(joined_path, columns=["item_id", "benchmark", "condition", "item_content"])
    rows = table.to_pylist()

    seen: dict[str, str] = {}
    for row in rows:
        iid = "" if row.get("item_id") is None else str(row.get("item_id"))
        if iid not in seen:
            seen[iid] = build_item_feature_text(row, max_chars=max_chars)
    print(f"  unique items: {len(seen):,}")

    item_id_order = list(seen.keys())
    texts = [seen[iid] for iid in item_id_order]

    is_dummy = encoder == "dummy"
    if is_dummy:
        dim = dummy_dim
        print(f"Encoding {len(texts):,} items with deterministic dummy encoder dim={dim} ...")
        emb = np.stack([_dummy_embedding(iid, dim) for iid in item_id_order]).astype(np.float32)
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
                "n_items": len(item_id_order),
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
    parser.add_argument("--feature-text-version", default=FEATURE_TEXT_VERSION)
    parser.add_argument("--dummy-dim", type=int, default=8, help="Embedding dimension when --encoder dummy is used.")
    args = parser.parse_args()
    main(args.joined, args.out, args.encoder, args.batch, args.max_chars, args.feature_text_version, args.dummy_dim)
