# Next Implementation Sprint Plan

## A. Syntax-heavy

Focus on `asyndetic_dash`, `consequence_dash`, `semicolon`, and detached constructions only after a bounded syntax provider path exists. These are suitable for the next diploma iteration only when probes can be candidate-backed and guarded by clean hard negatives.

## B. Dictionary-heavy

Prioritize `difficult_root_vowels`, then `suffix_vowels` and `verb_endings`. Use lexicon/morphology-backed candidates only; no generated correction enters training unless `CandidateGenerator` can reproduce it.

## C. NER-heavy

`capitalization_ner` and `typos_initials_period` need confirmed span support. `detached_applications` remains high-risk because it combines NER and syntax.

## D. CandidateGenerator Missing

After this sprint, `quote_pair_balance`, `bracket_pair_balance`, `punctuation_delete_replace`, `capitalization_sentence_start`, and `abbreviation_case_protection` have bounded candidate paths but are not `READY_NEXT_DATASET`. `quote_open`, `quote_close`, `yo_e_candidate`, and `neural_punctuation` remain inactive/missing for Phase 3 purposes.

## E. Validator

The next useful fix set is validator-focused: `abbreviation_case_protection`, `capitalization_sentence_start`, `dictionary_fuzzy`, `final_punctuation_default`, and `keyboard_typo_candidate`. These are suitable for the next diploma iteration because Phase 3 already shows candidate paths or existing positives but flags safety.

See `next_implementation_sprint_plan.csv` for per-rule implementation ideas, hard-negative ideas, risk, and priority.
