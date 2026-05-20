# Real Error Source Report

- downloads_enabled: yes
- dataset_loader_method: mixed_local_hf_jsonl
- accepted_pairs: 1021
- rejected_pairs: 16951
- rejection_reasons: {"candidate_missing": 3353, "char_edit_distance_too_high": 28, "code_or_social_marker": 24, "disallowed_error_type": 2431, "forbidden_domain_noise": 487, "low_cyrillic_ratio": 174, "markup_or_code_fragment": 270, "source_equals_target": 119, "source_share_cap": 625, "token_edit_distance_too_high": 599, "too_few_tokens": 1315, "too_many_tokens": 2048, "unsupported_edit_type": 5478}
- candidate_present_rate: 1.0000
- cap_rejections: 625
- final_verdict: READY_FOR_REBUILD_SOURCES

| source | status | mode | path | url/hf | bytes | seen | accepted | rejected | cap_rejections | reason | candidate_present_rate |
|---|---|---|---|---|---:|---:|---:|---:|---:|---|---:|
| spellcheck_benchmark | skipped | skipped |  | ai-forever/spellcheck_benchmark | 0 | 0 | 0 | 0 | 0 | materialized_to_sage_local_jsonl | 0.0000 |
| spellcheck_punctuation_benchmark | loaded | local | data/external/sage/spellcheck_punctuation_benchmark.jsonl | ai-forever/spellcheck_punctuation_benchmark | 9418333 | 8000 | 306 | 7694 | 529 |  | 1.0000 |
| sage_ruspellru | loaded | local | data/external/sage/RUSpellRU.jsonl |  | 1560484 | 3190 | 306 | 2884 | 96 |  | 1.0000 |
| sage_multidomain_gold | loaded | local | data/external/sage/MultidomainGold.jsonl |  | 5114317 | 5224 | 320 | 4837 | 0 |  | 1.0000 |
| sage_medspellchecker | loaded | local | data/external/sage/MedSpellChecker.jsonl |  | 563394 | 760 | 54 | 706 | 0 |  | 1.0000 |
| sage_github_typo_ru | loaded | local | data/external/sage/GitHubTypoCorpusRu.jsonl |  | 648735 | 865 | 35 | 830 | 0 |  | 1.0000 |
| rulec_gec | skipped | skipped | data/external/rulec_gec/ |  | 0 | 0 | 0 | 0 | 0 | disabled | 0.0000 |
