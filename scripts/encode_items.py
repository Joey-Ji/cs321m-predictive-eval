"""Stage 2a (part 1): encode every unique training item with a sentence transformer.

Reads data/joined.parquet (or data/items.parquet directly).
Writes:
  data/embeddings/item_embeddings.npy   — float32 [n_items, d]
  data/embeddings/item_id_order.json    — list[str] indexing rows of the npy
  data/embeddings/encoder_meta.json     — encoder name, dim, count

Default encoder: sentence-transformers/all-mpnet-base-v2 (768-d, fast).
Upgrade options: BAAI/bge-large-en-v1.5 (1024-d, MTEB 64.23).

Usage:
    python scripts/encode_items.py
    python scripts/encode_items.py --encoder BAAI/bge-large-en-v1.5 --batch 64
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(joined_path: Path, out_dir: Path, encoder: str, batch: int, max_chars: int) -> None:
    import numpy as np
    import pyarrow.parquet as pq
    from sentence_transformers import SentenceTransformer

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading {joined_path} ...")
    table = pq.read_table(joined_path, columns=["item_id", "item_content"])
    item_ids = table.column("item_id").to_pylist()
    item_texts = table.column("item_content").to_pylist()

    seen: dict[str, str] = {}
    for iid, text in zip(item_ids, item_texts):
        if iid not in seen:
            seen[iid] = (text or "")[:max_chars]
    print(f"  unique items: {len(seen):,}")

    item_id_order = list(seen.keys())
    texts = [seen[iid] for iid in item_id_order]

    print(f"Loading encoder {encoder} ...")
    model = SentenceTransformer(encoder)
    dim = model.get_sentence_embedding_dimension()
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
        json.dumps({"encoder": encoder, "dim": dim, "n_items": len(item_id_order), "max_chars": max_chars}, indent=2)
    )
    print(f"Done. Wrote {emb.shape} to {out_dir}/item_embeddings.npy")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--joined", default="data/joined.parquet", type=Path)
    parser.add_argument("--out", default="data/embeddings", type=Path)
    parser.add_argument("--encoder", default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--max-chars", type=int, default=4000, help="Truncate item text to this many chars before encoding.")
    args = parser.parse_args()
    main(args.joined, args.out, args.encoder, args.batch, args.max_chars)
