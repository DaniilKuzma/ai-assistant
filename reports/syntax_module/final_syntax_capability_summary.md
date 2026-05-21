# Final Syntax Capability Summary

This is an audit-only report. It does not build datasets, run training, change thresholds, or change checkpoints.

## Summary

- total_syntax_required_entries: 45
- syntax_required metadata/group entries: 1
- actionable syntax_required leaf entries: 44
- syntax_surface_total: 150
- syntax_surface_covered: 106
- syntax_surface_coverage_percent: 70.67
- current_syntax_required_total: 45
- current_syntax_required_covered: 0
- current_syntax_required_coverage_percent: 0.00
- syntax_plus_NER_blocked_count: 12
- syntax punctuation covered count: 82
- syntax orthography covered count: 24
- unsafe hard-negative accepted count: 0
- hard-negative overcorrection max: 0.0000

NER-required entries excluded from syntax coverage denominator; they are listed as separate blockers for a future NER/gazetteer module.

## Remaining Blockers By Reason

- needs_dictionary: 37
- needs_semantic_model: 6
- unsafe: 1

## Syntax Plus NER Blocked Entries

- orthography_4_1_1
- orthography_4_2_1
- orthography_4_2_4
- orthography_4_2_5
- orthography_4_2_6
- orthography_4_2_7
- orthography_4_6_1
- orthography_4_6_3
- orthography_4_7_1
- orthography_4_8
- orthography_4_9
- orthography_7_9

Verdict: SYNTAX_CAPABILITY_READY_FOR_DATASET
