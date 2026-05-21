# Bounded Candidate Implementation Report

## Scope

WORKING_V1 checkpoint and `calibrated_val_guarded_v3` were not modified. Orfogrammka orthography/punctuation pages were used only as taxonomy reference, not as a source for broad regex rule expansion.

## Implemented

| rule_id | implementation | mode | Phase 3 status | notes |
|---|---|---|---|---|
| quote_pair_balance | inserts only missing `«` or `»` when an opposite quote exists and boundary is clear | model_required | NEEDS_THRESHOLD | no straight quote normalization |
| bracket_pair_balance | inserts only missing opening/closing bracket when a one-sided pair exists | model_required | NEEDS_THRESHOLD | validator rejects increased imbalance |
| punctuation_delete_replace | duplicate/noise cleanup candidates for `,,`, `..`, `!!?`, `??!`, repeated `;`/`:` | candidate_only | NEEDS_THRESHOLD | no plain fallback auto-apply |
| capitalization_sentence_start | stricter sentence-start guards and no plain fallback auto-apply | model_required | NEEDS_VALIDATOR | blocks abbreviations, ellipsis middle, numeric/hyphen contexts, protected spans |
| abbreviation_case_protection | limited known-uppercase abbreviation candidates plus protected-span behavior | model_required | NEEDS_VALIDATOR | no aggressive unknown abbreviation correction |

## Kept Inactive

| rule_id | reason |
|---|---|
| quote_open | explicit broad normalization remains disabled; missing-pair repair is represented by `quote_pair_balance` |
| quote_close | explicit broad normalization remains disabled; missing-pair repair is represented by `quote_pair_balance` |
| yo_e_candidate | remains disabled unless `dictionary.yo_e.enabled=true` |
| neural_punctuation | remains metadata-only/planned |

## Validation

Targeted bounded tests passed. Matrix Eval Phase 3 now contains candidate-backed rows for the newly bounded rules where safe probes exist; none of these new bounded rules reached `READY_NEXT_DATASET`, so they are excluded from v3 training targets.
