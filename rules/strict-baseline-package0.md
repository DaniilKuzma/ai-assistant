# Package 0: Strict Error Baseline

Date: 2026-05-08
Branch: `main_transformer`

This tracked baseline freezes the current state before removing typo-like
augmentation and moving the project to a strict orthography + punctuation
scope.

Local generated JSON snapshot:
`report/strict_baseline_package0/baseline.json`

Note: `report/` is ignored by git, so the JSON file is a local reproducible
snapshot. This Markdown file is the tracked package-0 checkpoint.

## Source Reports

- `report/synthetic_v11/summary.json`
- `report/external_ai_forever_v11/summary.json`
- `report/external_ruspellgold_v11/summary.json`
- `report/*/error_metrics_by_type.csv`
- `data/processed/dataset.csv`
- `data/processed/train.csv`
- `data/processed/val.csv`
- `data/processed/test.csv`

## Baseline Metrics

| evaluation | examples | exact_match | CER delta | improved | worse | unchanged_wrong | punct_delta | mean_punct_count_error | accepted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| synthetic_v11 | 5,931 | 0.5709 | 0.0038 | 0.2627 | 0.0040 | 0.5867 | 0.0260 | 0.4434 | 0.9302 |
| external_ai_forever_v11 | 6,119 | 0.0431 | 0.0052 | 0.2955 | 0.0431 | 0.6387 | 0.0719 | 2.3769 | 0.9806 |
| external_ruspellgold_v11 | 713 | 0.1697 | 0.0018 | 0.2581 | 0.1038 | 0.6087 | -0.0124 | 0.7153 | 0.9776 |

Interpretation:

- Synthetic exact match is high mainly because the synthetic split contains many
  clean rows and simple local corruption patterns.
- External AI Forever and RuSpellGold remain hard: exact match is low and
  `unchanged_wrong_rate` is still around 0.61-0.64.
- RuSpellGold punctuation gets slightly worse on average (`punct_delta=-0.0124`),
  so later punctuation changes must be checked against this baseline.

## Current Dataset Distribution

Top current `data/processed/dataset.csv` error labels:

| error_type | count | strict decision |
| --- | ---: | --- |
| clean | 70,513 | neutral |
| spelling_replace | 37,562 | quarantine: too broad, split into real rules |
| punct_bracket_extra | 30,786 | quarantine/rewrite: currently too artificial |
| spelling_double | 22,659 | quarantine: keep only rule-based `н/нн` and double-consonant cases |
| spelling_extra | 22,405 | forbidden: typo-like random insertion |
| spelling_delete | 19,044 | forbidden: typo-like random deletion |
| punct_extra_comma | 17,849 | quarantine: keep only rule-based comma insertion cases |
| real_spelling | 16,151 | allowed after strict external-pair filtering |
| real_punctuation | 12,093 | allowed after strict external-pair filtering |
| punct_remove_final | 11,188 | allowed punctuation |
| spelling_swap | 9,198 | forbidden: typo-like transposition |

## Strict Taxonomy For Next Packages

Allowed orthography labels already present:

- `spelling_tsya`
- `spelling_suffix_pronunciation`
- `spelling_compound_joining`
- `spelling_capitalization`
- `spelling_abbreviation_case`
- `spelling_borrowed_word`
- `real_spelling`

Allowed punctuation labels already present:

- comma errors: `punct_remove_comma`, `punct_remove_internal_comma`,
  `punct_remove_comma_before_clause`, `punct_remove_intro_comma`,
  `punct_remove_all_commas`, selected rule-based `punct_extra_comma`
- final punctuation: `punct_remove_final`, `punct_remove_period`,
  `punct_remove_question`, `punct_remove_exclamation`
- wrong sign: `punct_wrong_period_to_comma`, `punct_wrong_comma_to_period`,
  `punct_wrong_colon_to_semicolon`, `punct_wrong_semicolon_to_colon`,
  `punct_wrong_question_to_period`, `punct_wrong_exclamation_to_period`
- structural punctuation: `punct_dash_missing`, `punct_dash_extra`,
  `punct_bracket_missing_close`, `punct_remove_quotes`,
  `punct_quote_punct_order`, `punct_direct_speech_inner_punct`,
  `punct_direct_speech_dash_missing`, `punct_list_missing_colon`,
  `punct_list_item_punctuation`, `punct_ellipsis_extra`,
  `punct_ellipsis_missing`
- `real_punctuation`

Quarantine labels that need rewrite or finer splitting:

- `spelling_replace`
- `spelling_double`
- `punct_extra_comma`
- `punct_bracket_extra`
- `punct_quote_style`

Forbidden for the strict scope:

- `spelling_delete`
- `spelling_swap`
- `spelling_extra`
- random `spelling_double`
- `real_other`
- `unknown`
- `keyboard`
- `spacing_only`
- `heavy_rewrite`

## Package 0 Acceptance Criteria

- Baseline summary metrics are captured.
- Current `error_types` distribution is captured.
- Allowed/quarantine/forbidden labels are explicitly listed.
- No model, dataset, or notebook logic is changed by this package.

