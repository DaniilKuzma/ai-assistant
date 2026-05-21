# Real Error Source Report

- downloads_enabled: no
- dataset_loader_method: mixed_local_hf_jsonl
- accepted_pairs: 3872
- rejected_pairs: 13667
- rejection_reasons: {"candidate_missing": 2571, "char_edit_distance_too_high": 28, "code_or_social_marker": 24, "disallowed_error_type": 2432, "forbidden_domain_noise": 487, "low_cyrillic_ratio": 174, "markup_or_code_fragment": 270, "source_equals_target": 119, "source_share_cap": 624, "token_edit_distance_too_high": 599, "too_few_tokens": 1315, "too_many_tokens": 2048, "unsupported_edit_type": 2976}
- candidate_present_rate: 1.0000
- cap_rejections: 624
- final_verdict: READY_FOR_REBUILD_SOURCES

| source | status | mode | path | url/hf | bytes | seen | accepted | rejected | cap_rejections | reason | candidate_present_rate |
|---|---|---|---|---|---:|---:|---:|---:|---:|---|---:|
| spellcheck_benchmark | skipped | skipped |  | ai-forever/spellcheck_benchmark | 0 | 0 | 0 | 0 | 0 | materialized_to_sage_local_jsonl | 0.0000 |
| spellcheck_punctuation_benchmark | loaded | local | data/external/sage/spellcheck_punctuation_benchmark.jsonl | ai-forever/spellcheck_punctuation_benchmark | 9418333 | 8000 | 1161 | 6839 | 624 |  | 1.0000 |
| sage_ruspellru | loaded | local | data/external/sage/RUSpellRU.jsonl |  | 1560484 | 3190 | 1144 | 2045 | 0 |  | 1.0000 |
| sage_multidomain_gold | loaded | local | data/external/sage/MultidomainGold.jsonl |  | 5114317 | 5224 | 1318 | 3407 | 0 |  | 1.0000 |
| sage_medspellchecker | loaded | local | data/external/sage/MedSpellChecker.jsonl |  | 563394 | 760 | 109 | 651 | 0 |  | 1.0000 |
| sage_github_typo_ru | loaded | local | data/external/sage/GitHubTypoCorpusRu.jsonl |  | 648735 | 865 | 140 | 725 | 0 |  | 1.0000 |
| rulec_gec | skipped | skipped | data/external/rulec_gec/ |  | 0 | 0 | 0 | 0 | 0 | disabled | 0.0000 |
