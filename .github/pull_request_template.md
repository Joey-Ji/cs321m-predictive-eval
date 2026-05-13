## What changed

<!-- One-line summary of the change. -->

## Why

<!-- Brief motivation. Reference the task in plan/calendar if applicable. -->

## Validation done

- [ ] `make test` passes (Stage 1 ↔ Stage 2 contract check)
- [ ] `make smoke-test SUB=submissions/<name>` passes (if a submission was touched)
- [ ] Offline validation log-likelihood improves vs current best (if this is a Codabench submission)

## Contract-breaking change?

The Stage 1 ↔ Stage 2 contract is the file schema written by `scripts/fit_irt.py` / `scripts/mock_irt.py` and consumed by `scripts/train_content_head.py` + `submissions/v1_irt/model.py`. Specifically: shapes of `data/irt/b.pt`, `data/irt/theta.pt`, `data/irt/log_a.pt`, and the keys in `data/irt/{subject,item}_to_id.json`.

- [ ] No — touches only one stage; the file schema above is unchanged.
- [ ] Yes — needs explicit sign-off from the other stage's owner. Briefly describe what breaks below.

## Notes

<!-- Anything reviewers should focus on. -->
