# Feature Build Profile Summary

- total rows profiled: 1
- total time: 3.022s
- rows/sec: 0.331
- p50/p90/p99 total_ms: 3020.796 / 3020.796 / 3020.796
- p50/p90/p99 candidate_generation_ms: 3016.259 / 3016.259 / 3016.259
- p50/p90/p99 dictionary_candidate_ms: 3011.493 / 3011.493 / 3011.493
- average candidates per row: 13.000
- average dictionary candidates per row: 4.000
- syntax enabled/disabled: disabled
- dictionary lexicon size: 3063822
- dictionary cache hit/miss stats: {'hits': 0, 'misses': 2, 'size': 2, 'max_size': 200000, 'hit_rate': 0.0}
- feature cache hit/miss stats: {'enabled': True, 'hit': False}
- clear bottleneck conclusion: dictionary dominates average row time (3011.493 ms)

## Top 50 Slowest Rows

- row_index=0 total_ms=3020.796 candidate_generation_ms=3016.259 dictionary_candidate_ms=3011.493 rule_ids=double_consonant_candidate|extra_letter_candidate|final_punctuation_default
