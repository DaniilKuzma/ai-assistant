# Short Dataset Generation Report

- verdict: READY_FOR_SHORT_TRAINING_DATASET
- total: 60000
- split_sizes: {"test": 5000, "train": 50000, "val": 5000}
- source_composition: {"clean": 3000, "hard_negative": 3000, "real": 1500, "synthetic": 52500}
- active_rule_ids: 59
- excluded_rule_ids: abbreviation_case_protection, asyndetic_dash, bracket_pair_balance, capitalization_ner, capitalization_sentence_start, consequence_dash, delete_replace, explanation_colon, neural_punctuation, punctuation_delete_replace, punctuation_noise, quote_close, quote_open, quote_pair_balance, quotes_brackets, semicolon, yo_e_candidate
- per_rule_count_min: 76
- per_rule_count_max: 6318
- max_dominance_rule_id: frequent_error_exact (5265)
- max_dominance_error_type: spelling (17496)
- candidate_recall_active_min: 1.000000
- candidate_recall_active_mean: 1.000000
- gap_coverage_active_min: 1.000000
- gap_coverage_active_mean: 1.000000
- hard_negative_count: 3000
- low_count_active_rule_ids: 

## Error Type Counts

{
  "clean_identity": 3000,
  "final_punctuation": 2500,
  "hard_negative": 3000,
  "hyphen": 6000,
  "punctuation": 15500,
  "spelling": 21000,
  "split_join": 9000
}

## Top Rule Counts

- frequent_error_exact: 6318
- comma_subordinate: 2800
- final_punctuation_default: 2500
- direct_speech_quotes: 2400
- comma_conjunction: 2200
- hyphen_whitelist: 2133
- ne_verb: 1600
- introductory_comma: 1500
- subject_predicate_dash: 1400
- detached_adverbial_comma: 1300
- homogeneous_comma: 1300
- address_comma: 1200
- direct_speech_colon: 1200
- enumeration_colon: 1200
- hyphen_particles: 1200
- context_nesmotrya: 1084
- comparative_turnover_comma: 1000
- hyphen_koe_koy: 1000
- hyphen_po_adverbs: 1000
- pol_polu_compounds: 1000
- pattern_цы_ци: 919
- ne_adjective: 889
- dictionary_fuzzy: 829
- double_consonant_candidate: 829
- extra_letter_candidate: 829
- keyboard_typo_candidate: 829
- missing_letter_candidate: 829
- prefix_pre_pri: 829
- swapped_letters_candidate: 829
- n_nn_adjective: 782
