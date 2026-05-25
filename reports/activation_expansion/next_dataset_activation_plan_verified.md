# Next Dataset Activation Plan Verified

Matrix READY_NEXT_DATASET count: 12
Verified Activation INCLUDE count: 11
Resolved mismatch: hyphen_whitelist

The verified activation list keeps only candidate-backed, sufficiently covered rules active for the next dataset cycle.

| rule_id | decision | status | recall | f1 | reason |
|---|---:|---|---:|---:|---|
| hyphen_whitelist | EXCLUDE | READY_NEXT_DATASET | 1.0 | 1.0 | under quota; Matrix READY_NEXT_DATASET but eval_count=1, KEEP_UNDER_QUOTA |
| address_comma | INCLUDE | READY_NEXT_DATASET | 1.0 | 1.0 | passes matrix gates |
| comma_conjunction | INCLUDE | READY_NEXT_DATASET | 1.0 | 0.924731182795699 | passes matrix gates |
| comma_subordinate | INCLUDE | READY_NEXT_DATASET | 1.0 | 1.0 | passes matrix gates |
| comparative_turnover_comma | INCLUDE | READY_NEXT_DATASET | 1.0 | 1.0 | passes matrix gates |
| detached_adverbial_comma | INCLUDE | READY_NEXT_DATASET | 1.0 | 1.0 | passes matrix gates |
| direct_speech_dash | INCLUDE | READY_NEXT_DATASET | 1.0 | 1.0 | passes matrix gates |
| homogeneous_comma | INCLUDE | READY_NEXT_DATASET | 1.0 | 0.99009900990099 | passes matrix gates |
| hyphen_koe_koy | INCLUDE | READY_NEXT_DATASET | 1.0 | 0.8190476190476191 | passes matrix gates |
| hyphen_particles | INCLUDE | READY_NEXT_DATASET | 1.0 | 0.761904761904762 | passes matrix gates |
| introductory_comma | INCLUDE | READY_NEXT_DATASET | 1.0 | 0.979591836734694 | passes matrix gates |
| subject_predicate_dash | INCLUDE | READY_NEXT_DATASET | 1.0 | 0.7816091954022989 | passes matrix gates |
