# Wave 1 Activation Plan Verified

Phase 2 READY_NEXT_DATASET count: 12
Verified Wave 1 INCLUDE count: 11
Resolved mismatch: hyphen_whitelist

The verified activation list keeps only candidate-backed, sufficiently covered rules active for the next dataset cycle.

| rule_id | decision | status | recall | f1 | reason |
|---|---:|---|---:|---:|---|
| hyphen_whitelist | EXCLUDE | READY_NEXT_DATASET | 1.0 | 1.0 | under quota; Phase 2 READY_NEXT_DATASET but eval_count=1, KEEP_UNDER_QUOTA |
| address_comma | INCLUDE | READY_NEXT_DATASET | 1.0 | 1.0 | passes phase2 gates |
| comma_conjunction | INCLUDE | READY_NEXT_DATASET | 1.0 | 0.924731182795699 | passes phase2 gates |
| comma_subordinate | INCLUDE | READY_NEXT_DATASET | 1.0 | 1.0 | passes phase2 gates |
| comparative_turnover_comma | INCLUDE | READY_NEXT_DATASET | 1.0 | 1.0 | passes phase2 gates |
| detached_adverbial_comma | INCLUDE | READY_NEXT_DATASET | 1.0 | 1.0 | passes phase2 gates |
| direct_speech_dash | INCLUDE | READY_NEXT_DATASET | 1.0 | 1.0 | passes phase2 gates |
| homogeneous_comma | INCLUDE | READY_NEXT_DATASET | 1.0 | 0.99009900990099 | passes phase2 gates |
| hyphen_koe_koy | INCLUDE | READY_NEXT_DATASET | 1.0 | 0.8190476190476191 | passes phase2 gates |
| hyphen_particles | INCLUDE | READY_NEXT_DATASET | 1.0 | 0.761904761904762 | passes phase2 gates |
| introductory_comma | INCLUDE | READY_NEXT_DATASET | 1.0 | 0.979591836734694 | passes phase2 gates |
| subject_predicate_dash | INCLUDE | READY_NEXT_DATASET | 1.0 | 0.7816091954022989 | passes phase2 gates |
