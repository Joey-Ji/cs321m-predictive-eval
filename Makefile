.PHONY: help install data eda irt irt-2pl irt-mock irt-mock-2pl encode encode-bge head head-mlp head-2pl submission smoke-test test test-quick clean clean-state list

# Default encoder for encode_items.py; override with `make encode ENCODER=...`
ENCODER ?= sentence-transformers/all-mpnet-base-v2

help:
	@echo "eval_comp workflow targets"
	@echo ""
	@echo "  Setup:"
	@echo "    make install              uv sync (install deps)"
	@echo ""
	@echo "  Data:"
	@echo "    make data                 Download HF dataset -> data/joined.parquet"
	@echo "    make eda                  Print one-page data summary"
	@echo ""
	@echo "  Stage 1 (IRT):"
	@echo "    make irt                  Fit Rasch (1PL) IRT -> data/irt/"
	@echo "    make irt-2pl              Fit 2PL IRT (predicts b AND log_a)"
	@echo "    make irt-mock             Generate synthetic Stage 1 outputs (1PL) for parallel dev"
	@echo "    make irt-mock-2pl         Same but with nonzero log_a (for 2PL development)"
	@echo ""
	@echo "  Stage 2a (content head):"
	@echo "    make encode               Encode items with all-mpnet-base-v2 -> data/embeddings/"
	@echo "    make encode-bge           Encode items with bge-large-en-v1.5 (upgrade)"
	@echo "    make head                 Train linear head, --targets b (1PL)"
	@echo "    make head-mlp             Train MLP head, --targets b"
	@echo "    make head-2pl             Train MLP head, --targets b+log_a (2PL)"
	@echo ""
	@echo "  Submissions:"
	@echo "    make submission NAME=v1_irt          Build submissions/v1_irt.zip"
	@echo "    make smoke-test SUB=submissions/v1_irt   CPU-test predict() locally"
	@echo ""
	@echo "  Testing:"
	@echo "    make test                 Run contract tests (uses mock data)"
	@echo ""
	@echo "  Cleanup:"
	@echo "    make clean                Remove __pycache__ and *.egg-info"
	@echo "    make clean-state          Remove data/irt, data/embeddings, data/head (KEEPS joined.parquet)"
	@echo ""
	@echo "  Discovery:"
	@echo "    make list                 List submission dirs and built ZIPs"

install:
	uv sync

data:
	uv run python scripts/download_data.py

eda:
	uv run python scripts/eda.py

irt:
	uv run python scripts/fit_irt.py --model 1pl

irt-2pl:
	uv run python scripts/fit_irt.py --model 2pl --epochs 300

irt-mock:
	uv run python scripts/mock_irt.py --model 1pl

irt-mock-2pl:
	uv run python scripts/mock_irt.py --model 2pl

encode:
	uv run python scripts/encode_items.py --encoder $(ENCODER)

encode-bge:
	uv run python scripts/encode_items.py --encoder BAAI/bge-large-en-v1.5

head:
	uv run python scripts/train_content_head.py --head linear --targets b

head-mlp:
	uv run python scripts/train_content_head.py --head mlp --targets b --epochs 200

head-2pl:
	uv run python scripts/train_content_head.py --head mlp --targets b+log_a --epochs 200

submission:
	@if [ -z "$(NAME)" ]; then echo "ERROR: usage: make submission NAME=v1_irt"; exit 1; fi
	uv run python scripts/build_submission.py $(NAME)

smoke-test:
	@if [ -z "$(SUB)" ]; then echo "ERROR: usage: make smoke-test SUB=submissions/v1_irt"; exit 1; fi
	uv run python scripts/smoke_test.py $(SUB)

test:
	uv run python tests/check_contract.py

list:
	@echo "Submission directories:"
	@ls -d submissions/*/ 2>/dev/null | sed 's|^|  |' || echo "  (none)"
	@echo ""
	@echo "Built ZIPs:"
	@ls -lh submissions/*.zip 2>/dev/null | awk '{print "  " $$NF, "(" $$5 ")"}' || echo "  (none)"

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

clean-state:
	rm -rf data/irt data/embeddings data/head
	@echo "Removed stage outputs. data/joined.parquet KEPT."
