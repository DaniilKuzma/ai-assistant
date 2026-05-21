# Validator Fix Report

Phase 2 validator fixes are bounded safety guards only; thresholds and checkpoints are unchanged.

| rule_id | decision | accepted | rejected | top_rejections | note |
|---|---:|---:|---:|---|---|
| double_consonant_candidate | NEEDS_THRESHOLD | 0 | 10 | unsafe_double_consonant_candidate:10 | fixed or safely blocked after guard update |
| extra_letter_candidate | NEEDS_THRESHOLD | 0 | 3 | known_source_lexical_guard:2; protected_lexical_guard:1 | fixed or safely blocked after guard update |
| hyphen_po_adverbs | NEEDS_THRESHOLD | 0 | 10 | unsafe_hyphen_po_adverb:10 | fixed or safely blocked after guard update |
| missing_letter_candidate | NEEDS_THRESHOLD | 0 | 1 | known_source_lexical_guard:1 | fixed or safely blocked after guard update |
| pattern_чо_че | NEEDS_THRESHOLD | 0 | 1 | known_source_lexical_guard:1 | fixed or safely blocked after guard update |
