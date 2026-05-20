# Item Residual Phase 0

- joined rows: 4443797
- items: 70873
- folds: 5 seed=0
- lambda_item: 10.0
- kappas: `{"benchmark": 2.0, "benchmark_condition": 2.0, "subject": 2.0, "subject_benchmark": 2.0, "subject_category": 2.0}`

## Target Distribution

- mean(delta): 0.023541
- std(delta): 0.293253
- min/max(delta): -0.500000 / 0.500000
- |delta| > 0.25: 27088
- clipped at +/-0.50: 19116

## Decision

- proceed_to_phase1: `True`
- reason: target std is large enough to attempt a content model

## Leakage Spot Check

- item `00003145688411a8` fold=2 train_contains_item=False
- item `000145f2b81378e4` fold=0 train_contains_item=False
- item `0001783c205f1448` fold=0 train_contains_item=False
- item `00030460ff2b8e5a` fold=3 train_contains_item=False
- item `00042aaaa35e0f2a` fold=3 train_contains_item=False
- item `00068f04d48c91e0` fold=1 train_contains_item=False
- item `0006969cb1b21f5c` fold=0 train_contains_item=False
- item `00069d1681f4d04c` fold=1 train_contains_item=False
