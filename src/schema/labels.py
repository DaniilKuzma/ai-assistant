from __future__ import annotations

from enum import Enum
from types import MappingProxyType
from typing import Mapping


class TokenEditLabel(str, Enum):
    KEEP = "KEEP"
    DELETE = "DELETE"
    SKIP_MERGED = "SKIP_MERGED"
    LOWERCASE = "LOWERCASE"
    UPPERCASE = "UPPERCASE"
    SPLIT_NE_VERB = "SPLIT_NE_VERB"
    MERGE_TAK_ZHE_TO_TAKZHE = "MERGE_TAK_ZHE_TO_TAKZHE"
    SPLIT_TAKZHE_TO_TAK_ZHE = "SPLIT_TAKZHE_TO_TAK_ZHE"
    MERGE_TO_ZHE_TO_TOZHE = "MERGE_TO_ZHE_TO_TOZHE"
    SPLIT_TOZHE_TO_TO_ZHE = "SPLIT_TOZHE_TO_TO_ZHE"
    MERGE_ZA_TO_TO_ZATO = "MERGE_ZA_TO_TO_ZATO"
    SPLIT_ZATO_TO_ZA_TO = "SPLIT_ZATO_TO_ZA_TO"
    HYPHENATE_PARTICLE_TO = "HYPHENATE_PARTICLE_TO"
    HYPHENATE_PARTICLE_LIBO = "HYPHENATE_PARTICLE_LIBO"
    HYPHENATE_PARTICLE_NIBUD = "HYPHENATE_PARTICLE_NIBUD"
    HYPHENATE_KOE = "HYPHENATE_KOE"
    HYPHENATE_PO_ADVERB = "HYPHENATE_PO_ADVERB"
    FIX_TSYA_TO_TTSYA = "FIX_TSYA_TO_TTSYA"
    FIX_TTSYA_TO_TSYA = "FIX_TTSYA_TO_TSYA"
    DICT_REPLACE = "DICT_REPLACE"
    SPAN_REPLACE_BY_LEXICON = "SPAN_REPLACE_BY_LEXICON"
    CAPITALIZE = "CAPITALIZE"


class GapPunctuationLabel(str, Enum):
    NONE = "NONE"
    COMMA = "COMMA"
    DASH = "DASH"
    COLON = "COLON"
    SEMICOLON = "SEMICOLON"
    DOT = "DOT"
    QUESTION = "QUESTION"
    EXCLAMATION = "EXCLAMATION"
    ELLIPSIS = "ELLIPSIS"
    DELETE_PUNCTUATION = "DELETE_PUNCTUATION"
    COMMA_DASH = "COMMA_DASH"


class BoundaryBeforeLabel(str, Enum):
    NONE = "NONE"
    INSERT_OPEN_QUOTE = "INSERT_OPEN_QUOTE"
    DELETE_OPEN_QUOTE = "DELETE_OPEN_QUOTE"
    NORMALIZE_OPEN_QUOTE = "NORMALIZE_OPEN_QUOTE"
    INSERT_OPEN_BRACKET = "INSERT_OPEN_BRACKET"
    DELETE_OPEN_BRACKET = "DELETE_OPEN_BRACKET"


class BoundaryAfterLabel(str, Enum):
    NONE = "NONE"
    INSERT_CLOSE_QUOTE = "INSERT_CLOSE_QUOTE"
    DELETE_CLOSE_QUOTE = "DELETE_CLOSE_QUOTE"
    NORMALIZE_CLOSE_QUOTE = "NORMALIZE_CLOSE_QUOTE"
    INSERT_CLOSE_BRACKET = "INSERT_CLOSE_BRACKET"
    DELETE_CLOSE_BRACKET = "DELETE_CLOSE_BRACKET"


class RuleLabel(str, Enum):
    NONE = "none"
    CLEAN_IDENTITY = "clean_identity"
    NE_VERB = "ne_verb"
    TAKZHE_TAK_ZHE = "takzhe_tak_zhe"
    TOZHE_TO_ZHE = "tozhe_to_zhe"
    ZATO_ZA_TO = "zato_za_to"
    HYPHEN_PARTICLES = "hyphen_particles"
    HYPHEN_KOE = "hyphen_koe"
    HYPHEN_PO_ADVERB = "hyphen_po_adverb"
    TSYA_TTSYA = "tsya_ttsya"
    COMMA_SUBORDINATE = "comma_subordinate"
    COMMA_INTRODUCTORY = "comma_introductory"
    COMMA_HOMOGENEOUS = "comma_homogeneous"
    COMMA_ADVERSATIVE = "comma_adversative"
    DASH_SUBJECT_PREDICATE = "dash_subject_predicate"
    FINAL_PUNCTUATION = "final_punctuation"
    SUFFIX_ITS_ETS = "suffix_its_ets"
    SUFFIX_ENN_YAN = "suffix_enn_yan"
    N_NN_BASIC = "n_nn_basic"
    COMPOUND_SERVICE_WORDS = "compound_service_words"
    COMPOUND_PREPOSITIONS = "compound_prepositions"
    COMPOUND_PRONOUNS_PARTICLES = "compound_pronouns_particles"
    COMPOUND_ADVERBS = "compound_adverbs"
    COMPOUND_NOUNS_ADJECTIVES = "compound_nouns_adjectives"
    COMPOUND_NE_SPELLINGS = "compound_ne_spellings"
    COMPOUND_POL_POLU = "compound_pol_polu"
    MORPHEME_HISSING_VOWELS = "morpheme_hissing_vowels"
    MORPHEME_SOFT_HARD_SIGNS = "morpheme_soft_hard_signs"
    MORPHEME_ROOT_VOWELS = "morpheme_root_vowels"
    MORPHEME_PREFIXES = "morpheme_prefixes"
    MORPHEME_SUFFIXES = "morpheme_suffixes"
    MORPHEME_N_NN = "morpheme_n_nn"
    MORPHEME_CONSONANTS = "morpheme_consonants"
    MORPHEME_ENDINGS = "morpheme_endings"
    DICTIONARY_NORMATIVE_WORDS = "dictionary_normative_words"
    DICTIONARY_BORROWED_WORDS = "dictionary_borrowed_words"
    DICTIONARY_DOMAIN_TERMS = "dictionary_domain_terms"
    DICTIONARY_COMMON_MISSPELLINGS = "dictionary_common_misspellings"
    TYPO_CHARACTER_NOISE = "typo_character_noise"
    TYPO_KEYBOARD_NEIGHBOR = "typo_keyboard_neighbor"
    TYPO_SPACE_NOISE = "typo_space_noise"
    PUNCT_FINAL_MARKS = "punct_final_marks"
    PUNCT_DASH_SYNTAX = "punct_dash_syntax"
    PUNCT_HOMOGENEOUS_EXTENDED = "punct_homogeneous_extended"
    PUNCT_DETACHED_DEFINITIONS = "punct_detached_definitions"
    PUNCT_DETACHED_ADVERBIALS = "punct_detached_adverbials"
    PUNCT_COMPARATIVE_TURNS = "punct_comparative_turns"
    PUNCT_INTRODUCTORY_EXTENDED = "punct_introductory_extended"
    PUNCT_ADDRESS_INTERJECTION = "punct_address_interjection"
    PUNCT_COMPLEX_SENTENCES = "punct_complex_sentences"
    PUNCT_BSP = "punct_bsp"
    PUNCT_FIXED_EXPRESSION_GUARDS = "punct_fixed_expression_guards"
    QUOTE_PAIRING = "quote_pairing"
    QUOTE_NORMALIZATION = "quote_normalization"
    QUOTE_EXTRA_MARKS = "quote_extra_marks"
    DIALOGUE_AUTHOR_BEFORE = "dialogue_author_before"
    DIALOGUE_SPEECH_BEFORE_AUTHOR = "dialogue_speech_before_author"
    DIALOGUE_AUTHOR_INSIDE_SPEECH = "dialogue_author_inside_speech"
    DIALOGUE_BRACKET_GUARDS = "dialogue_bracket_guards"
    CASING_SENTENCE_START = "casing_sentence_start"
    CASING_PERSON_NAMES = "casing_person_names"
    CASING_GEO_NAMES = "casing_geo_names"
    CASING_ORGANIZATIONS = "casing_organizations"
    CASING_DOCUMENTS_EVENTS = "casing_documents_events"
    CASING_COMMON_LOWERCASE = "casing_common_lowercase"
    CASING_FORMAL_YOU_GUARD = "casing_formal_you_guard"
    SEMANTIC_SERVICE_WORDS = "semantic_service_words"
    SEMANTIC_DERIVED_PREPOSITIONS = "semantic_derived_prepositions"
    SEMANTIC_NE_NI = "semantic_ne_ni"
    SEMANTIC_INTRODUCTORY_CONTEXT = "semantic_introductory_context"
    SEMANTIC_COMPARATIVE_CONTEXT = "semantic_comparative_context"


TOKEN_EDIT_LABELS: tuple[str, ...] = (
    "KEEP",
    "DELETE",
    "SKIP_MERGED",
    "LOWERCASE",
    "UPPERCASE",
    "SPLIT_NE_VERB",
    "MERGE_TAK_ZHE_TO_TAKZHE",
    "SPLIT_TAKZHE_TO_TAK_ZHE",
    "MERGE_TO_ZHE_TO_TOZHE",
    "SPLIT_TOZHE_TO_TO_ZHE",
    "MERGE_ZA_TO_TO_ZATO",
    "SPLIT_ZATO_TO_ZA_TO",
    "HYPHENATE_PARTICLE_TO",
    "HYPHENATE_PARTICLE_LIBO",
    "HYPHENATE_PARTICLE_NIBUD",
    "HYPHENATE_KOE",
    "HYPHENATE_PO_ADVERB",
    "FIX_TSYA_TO_TTSYA",
    "FIX_TTSYA_TO_TSYA",
    "DICT_REPLACE",
    "SPAN_REPLACE_BY_LEXICON",
    "CAPITALIZE",
)

GAP_PUNCTUATION_LABELS: tuple[str, ...] = (
    "NONE",
    "COMMA",
    "DASH",
    "COLON",
    "SEMICOLON",
    "DOT",
    "QUESTION",
    "EXCLAMATION",
    "ELLIPSIS",
    "DELETE_PUNCTUATION",
    "COMMA_DASH",
)

BOUNDARY_BEFORE_LABELS: tuple[str, ...] = (
    "NONE",
    "INSERT_OPEN_QUOTE",
    "DELETE_OPEN_QUOTE",
    "NORMALIZE_OPEN_QUOTE",
    "INSERT_OPEN_BRACKET",
    "DELETE_OPEN_BRACKET",
)

BOUNDARY_AFTER_LABELS: tuple[str, ...] = (
    "NONE",
    "INSERT_CLOSE_QUOTE",
    "DELETE_CLOSE_QUOTE",
    "NORMALIZE_CLOSE_QUOTE",
    "INSERT_CLOSE_BRACKET",
    "DELETE_CLOSE_BRACKET",
)

RULE_LABELS: tuple[str, ...] = (
    "none",
    "clean_identity",
    "ne_verb",
    "takzhe_tak_zhe",
    "tozhe_to_zhe",
    "zato_za_to",
    "hyphen_particles",
    "hyphen_koe",
    "hyphen_po_adverb",
    "tsya_ttsya",
    "comma_subordinate",
    "comma_introductory",
    "comma_homogeneous",
    "comma_adversative",
    "dash_subject_predicate",
    "final_punctuation",
    "suffix_its_ets",
    "suffix_enn_yan",
    "n_nn_basic",
    "compound_service_words",
    "compound_prepositions",
    "compound_pronouns_particles",
    "compound_adverbs",
    "compound_nouns_adjectives",
    "compound_ne_spellings",
    "compound_pol_polu",
    "morpheme_hissing_vowels",
    "morpheme_soft_hard_signs",
    "morpheme_root_vowels",
    "morpheme_prefixes",
    "morpheme_suffixes",
    "morpheme_n_nn",
    "morpheme_consonants",
    "morpheme_endings",
    "dictionary_normative_words",
    "dictionary_borrowed_words",
    "dictionary_domain_terms",
    "dictionary_common_misspellings",
    "typo_character_noise",
    "typo_keyboard_neighbor",
    "typo_space_noise",
    "punct_final_marks",
    "punct_dash_syntax",
    "punct_homogeneous_extended",
    "punct_detached_definitions",
    "punct_detached_adverbials",
    "punct_comparative_turns",
    "punct_introductory_extended",
    "punct_address_interjection",
    "punct_complex_sentences",
    "punct_bsp",
    "punct_fixed_expression_guards",
    "quote_pairing",
    "quote_normalization",
    "quote_extra_marks",
    "dialogue_author_before",
    "dialogue_speech_before_author",
    "dialogue_author_inside_speech",
    "dialogue_bracket_guards",
    "casing_sentence_start",
    "casing_person_names",
    "casing_geo_names",
    "casing_organizations",
    "casing_documents_events",
    "casing_common_lowercase",
    "casing_formal_you_guard",
    "semantic_service_words",
    "semantic_derived_prepositions",
    "semantic_ne_ni",
    "semantic_introductory_context",
    "semantic_comparative_context",
)

TOKEN_LABEL_TO_ID: Mapping[str, int] = MappingProxyType(
    {label: index for index, label in enumerate(TOKEN_EDIT_LABELS)}
)
TOKEN_ID_TO_LABEL: tuple[str, ...] = TOKEN_EDIT_LABELS

GAP_LABEL_TO_ID: Mapping[str, int] = MappingProxyType(
    {label: index for index, label in enumerate(GAP_PUNCTUATION_LABELS)}
)
GAP_ID_TO_LABEL: tuple[str, ...] = GAP_PUNCTUATION_LABELS

BOUNDARY_BEFORE_LABEL_TO_ID: Mapping[str, int] = MappingProxyType(
    {label: index for index, label in enumerate(BOUNDARY_BEFORE_LABELS)}
)
BOUNDARY_BEFORE_ID_TO_LABEL: tuple[str, ...] = BOUNDARY_BEFORE_LABELS

BOUNDARY_AFTER_LABEL_TO_ID: Mapping[str, int] = MappingProxyType(
    {label: index for index, label in enumerate(BOUNDARY_AFTER_LABELS)}
)
BOUNDARY_AFTER_ID_TO_LABEL: tuple[str, ...] = BOUNDARY_AFTER_LABELS

RULE_LABEL_TO_ID: Mapping[str, int] = MappingProxyType(
    {label: index for index, label in enumerate(RULE_LABELS)}
)
RULE_ID_TO_LABEL: tuple[str, ...] = RULE_LABELS


def token_label_to_id(label: str) -> int:
    try:
        return TOKEN_LABEL_TO_ID[label]
    except KeyError as exc:
        raise ValueError(f"Unknown token edit label: {label!r}.") from exc


def token_id_to_label(id_: int) -> str:
    if not isinstance(id_, int) or id_ < 0 or id_ >= len(TOKEN_ID_TO_LABEL):
        raise ValueError(f"Unknown token edit id: {id_!r}.")
    return TOKEN_ID_TO_LABEL[id_]


def gap_label_to_id(label: str) -> int:
    try:
        return GAP_LABEL_TO_ID[label]
    except KeyError as exc:
        raise ValueError(f"Unknown gap punctuation label: {label!r}.") from exc


def gap_id_to_label(id_: int) -> str:
    if not isinstance(id_, int) or id_ < 0 or id_ >= len(GAP_ID_TO_LABEL):
        raise ValueError(f"Unknown gap punctuation id: {id_!r}.")
    return GAP_ID_TO_LABEL[id_]


def boundary_before_label_to_id(label: str) -> int:
    try:
        return BOUNDARY_BEFORE_LABEL_TO_ID[label]
    except KeyError as exc:
        raise ValueError(f"Unknown boundary-before label: {label!r}.") from exc


def boundary_before_id_to_label(id_: int) -> str:
    if not isinstance(id_, int) or id_ < 0 or id_ >= len(BOUNDARY_BEFORE_ID_TO_LABEL):
        raise ValueError(f"Unknown boundary-before id: {id_!r}.")
    return BOUNDARY_BEFORE_ID_TO_LABEL[id_]


def boundary_after_label_to_id(label: str) -> int:
    try:
        return BOUNDARY_AFTER_LABEL_TO_ID[label]
    except KeyError as exc:
        raise ValueError(f"Unknown boundary-after label: {label!r}.") from exc


def boundary_after_id_to_label(id_: int) -> str:
    if not isinstance(id_, int) or id_ < 0 or id_ >= len(BOUNDARY_AFTER_ID_TO_LABEL):
        raise ValueError(f"Unknown boundary-after id: {id_!r}.")
    return BOUNDARY_AFTER_ID_TO_LABEL[id_]


def rule_tag_to_id(label: str) -> int:
    try:
        return RULE_LABEL_TO_ID[label]
    except KeyError as exc:
        raise ValueError(f"Unknown rule label: {label!r}.") from exc


def rule_id_to_label(id_: int) -> str:
    if not isinstance(id_, int) or id_ < 0 or id_ >= len(RULE_ID_TO_LABEL):
        raise ValueError(f"Unknown rule id: {id_!r}.")
    return RULE_ID_TO_LABEL[id_]


__all__ = [
    "BOUNDARY_AFTER_ID_TO_LABEL",
    "BOUNDARY_AFTER_LABELS",
    "BOUNDARY_AFTER_LABEL_TO_ID",
    "BOUNDARY_BEFORE_ID_TO_LABEL",
    "BOUNDARY_BEFORE_LABELS",
    "BOUNDARY_BEFORE_LABEL_TO_ID",
    "BoundaryAfterLabel",
    "BoundaryBeforeLabel",
    "GAP_ID_TO_LABEL",
    "GAP_LABEL_TO_ID",
    "GAP_PUNCTUATION_LABELS",
    "GapPunctuationLabel",
    "RULE_ID_TO_LABEL",
    "RULE_LABELS",
    "RULE_LABEL_TO_ID",
    "RuleLabel",
    "TOKEN_EDIT_LABELS",
    "TOKEN_ID_TO_LABEL",
    "TOKEN_LABEL_TO_ID",
    "TokenEditLabel",
    "boundary_after_id_to_label",
    "boundary_after_label_to_id",
    "boundary_before_id_to_label",
    "boundary_before_label_to_id",
    "gap_id_to_label",
    "gap_label_to_id",
    "rule_id_to_label",
    "rule_tag_to_id",
    "token_id_to_label",
    "token_label_to_id",
]
