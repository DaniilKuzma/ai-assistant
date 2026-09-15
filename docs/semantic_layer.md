# Semantic RuleLayer

The `semantic` RuleLayer covers bounded contextual disambiguation for Russian
spelling and punctuation cases. It stays inside the existing AST-first online
generation and direct edit tagging architecture: examples emit token and gap
labels, and runtime span replacements remain lexicon-gated.

## Runtime Policy

Semantic corrections may be loaded into `OrthographicCorrectionLexicon`, but
deterministic lexicon correction does not trust `semantic_` rule ids. A semantic
replacement can run only through the neural direct-label path, where
`SPAN_REPLACE_BY_LEXICON` asks the lexicon for the exact grouped rule id.

Legacy-overlap service-word edits use specialized labels:

- `MERGE_TAK_ZHE_TO_TAKZHE` / `SPLIT_TAKZHE_TO_TAK_ZHE`
- `MERGE_TO_ZHE_TO_TOZHE` / `SPLIT_TOZHE_TO_TO_ZHE`
- `MERGE_ZA_TO_TO_ZATO` / `SPLIT_ZATO_TO_ZA_TO`

Generic semantic spans use `SPAN_REPLACE_BY_LEXICON` and entries in
`lexicon/layers/semantic/corrections.yaml`.

## Coverage

| File | Rule id | Sub-rule ids | Edits | Guards |
| --- | --- | --- | --- | --- |
| `service_words.yaml` | `semantic_service_words` | `takzhe_additive_merge`, `takzhe_comparison_split`, `tozhe_additive_merge`, `tozhe_demonstrative_split`, `zato_adversative_merge`, `zato_preposition_split` | Specialized merge/split labels for legacy overlap. | `takzhe_comparison_guard`, `tozhe_demonstrative_guard`, `zato_preposition_guard` |
| `service_words.yaml` | `semantic_service_words` | `chtoby_conjunction_merge`, `potomu_chto_merge`, `ottogo_merge`, `otchego_merge`, `zachem_merge` | `SPAN_REPLACE_BY_LEXICON` with semantic correction lexicon entries. | `chtoby_particle_split`, `potomu_chto_guard`, `ottogo_guard`, `otchego_guard`, `zachem_guard` |
| `derived_prepositions.yaml` | `semantic_derived_prepositions` | `v_techenie_temporal`, `vsledstvie_causal`, `naschet_about`, `vvidu_causal`, `nesmotrya_concessive` | `SPAN_REPLACE_BY_LEXICON` with semantic correction lexicon entries. | `v_techenie_literal_guard`, `vsledstvie_literal_guard`, `naschet_literal_guard`, `vvidu_literal_guard`, `nesmotrya_literal_guard` |
| `ne_ni_guards.yaml` | `semantic_ne_ni` | none | Guard-only. No word insertion such as adding `как`; hard-negative and clean-identity cases keep valid phrasing unchanged. | `ne_kto_inoy_guard`, `nikto_inoy_guard`, `ne_chto_inoe_guard`, `nichto_inoe_guard` |
| `introductory_context.yaml` | `semantic_introductory_context` | none | Guard-only to avoid duplicating `punct_introductory_extended` positives; hard-negative and clean-identity cases keep homonym contexts unchanged. | `introductory_homonym_guard` |
| `comparative_context.yaml` | `semantic_comparative_context` | `comparative_kak_appositive` | Two `COMMA` gap labels for safe appositive `как` contexts. | `comparative_kak_role_guard` |

Current expanded case counts:

| Rule id | Positive | Hard negative | Clean identity |
| --- | ---: | ---: | ---: |
| `semantic_service_words` | 261 | 32 | 0 |
| `semantic_derived_prepositions` | 141 | 20 | 0 |
| `semantic_ne_ni` | 0 | 16 | 4 |
| `semantic_introductory_context` | 0 | 5 | 2 |
| `semantic_comparative_context` | 52 | 4 | 0 |

## Metadata Contract

Every semantic case includes:

- `semantic_case_type`
- `ambiguity_pair`
- `semantic_signal`
- `operation`
- `source`
- `target`
- `legacy_overlap`
- `compound_overlap`

Every semantic `LayerRuleSpec` also declares `supports_positive`,
`supports_hard_negative`, `supports_clean_identity`, and `rule_kind`.
`semantic_ne_ni` and `semantic_introductory_context` are enabled guard-only
specs: `rule_kind` is `guard`, positive generation is unsupported, and
coverage comes from hard-negative plus clean-identity identity examples.

Allowed `semantic_case_type` values are `additive`, `comparison`,
`demonstrative`, `causal`, `temporal`, `literal`, `concession`, and `guard`.

## Out Of Scope

The semantic layer must not register legacy rule ids such as `takzhe_tak_zhe`,
`tozhe_to_zhe`, `zato_za_to`, or `ne_verb`. It also must not restore the old
candidate-aware dataset pipeline, create materialized train/val/test CSVs, add
seq2seq correction, or introduce free-form rewrite behavior.
