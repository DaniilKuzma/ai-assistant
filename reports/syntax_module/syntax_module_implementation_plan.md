# Syntax Module Implementation Plan

This is an implementation handoff for future syntax-module work. It does not build data, run training, change thresholds, or change model checkpoints.

## Baseline

- total syntax_required entries: 122
- total leaf syntax_required entries: 122
- families count: 25 total, 17 nonzero
- implement_now count: 1
- bounded_pattern count: 75
- blocked count: 46
- expected training_eligible gain: 76

## Files To Create

- `src/nlp/syntax_features.py`: dependency helpers over `SyntaxToken`, clause heads, finite verb detection, subtree spans, agreement, coordination chains, and protected gap checks.
- `src/rules/syntax_punctuation.py`: bounded candidate builders grouped by syntax punctuation family.
- `src/rules/syntax_orthography.py`: bounded syntax-aware split/join candidates for `не` and context-pair families.

## Candidate And Validator Changes

- Extend `Candidate` and `PunctuationGapCandidate` with optional `syntax_family`, `implementation_group`, `constraint_group`, and `bundle_id` fields.
- Update `CandidateGenerator` to parse syntax once per text and pass syntax features into punctuation and span candidate builders.
- Add `StrictValidator` guards per syntax family; keep every model-backed edit behind trusted candidates.
- Add decoder constraints: one punctuation action per gap, paired direct-speech/quote/bracket bundles, comma-pair bundles, and no conflicting punctuation operations.

## Implementation Families

- priority 1: `subordinate_clause_comma` covers 4 entries; candidate rule ids: comma_subordinate.
- priority 2: `coordinating_conjunction_comma` covers 1 entries; candidate rule ids: comma_conjunction.
- priority 1: `introductory_words` covers 8 entries; candidate rule ids: introductory_comma.
- priority 1: `homogeneous_members` covers 6 entries; candidate rule ids: homogeneous_comma, enumeration_colon.
- priority 1: `detached_adverbial_phrases` covers 8 entries; candidate rule ids: detached_adverbial_comma.
- priority 2: `detached_participial_phrases` covers 4 entries; candidate rule ids: detached_participial_comma.
- priority 4: `detached_applications` covers 1 entries; candidate rule ids: detached_application_comma.
- priority 2: `clarification_members` covers 3 entries; candidate rule ids: clarification_comma.
- priority 3: `comparative_turnovers` covers 5 entries; candidate rule ids: comparative_turnover_comma.
- priority 1: `subject_predicate_dash` covers 12 entries; candidate rule ids: subject_predicate_dash.
- priority 2: `enumeration_colon_dash` covers 1 entries; candidate rule ids: enumeration_colon.
- priority 2: `direct_speech_syntax` covers 4 entries; candidate rule ids: direct_speech_colon, direct_speech_dash, direct_speech_quotes.
- priority 3: `punctuation_combinations` covers 7 entries; candidate rule ids: punctuation_delete_replace.
- priority 3: `ne_with_parts_of_speech` covers 7 entries; candidate rule ids: ne_adjective, ne_adverb, ne_participle, ne_verb.
- priority 3: `context_pairs` covers 5 entries; candidate rule ids: context_chto_by, context_nesmotrya, context_tak_zhe, context_to_zhe, context_vsledstvie, context_za_to.

## Blocked Or Deferred

- `comparative_turnovers`: 2 blocked/deferred entries.
- `asyndetic_complex_sentence`: 1 blocked/deferred entries.
- `punctuation_combinations`: 11 blocked/deferred entries.
- `ne_with_parts_of_speech`: 8 blocked/deferred entries.
- `context_pairs`: 17 blocked/deferred entries.
- `grammatical_endings_context`: 7 blocked/deferred entries.

## Rules YAML Updates

- Update `configs/rules.yaml` only after real candidate builders and tests exist.
- Change implemented entries from `syntax_required` to `model_required`, set `executable: true`, add canonical `rule_ids`, and update dataset metadata.
- Do not add fake rules, do not change thresholds, and do not change checkpoints.

## Tests

- Add syntax feature unit tests for clause heads, finite verbs, subtree spans, agreement, coordination, and protected gaps.
- Add candidate tests for each implemented family with negative overcorrection examples.
- Add validator tests for risky punctuation, paired bundles, context pairs, and `не` split/join.
- Keep `python -m pytest -q tests/test_syntax_required_inventory.py` passing after every audit refresh.

SYNTAX_REQUIRED_AUDIT_COMPLETE
