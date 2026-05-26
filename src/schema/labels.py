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
)

TOKEN_LABEL_TO_ID: Mapping[str, int] = MappingProxyType(
    {label: index for index, label in enumerate(TOKEN_EDIT_LABELS)}
)
TOKEN_ID_TO_LABEL: tuple[str, ...] = TOKEN_EDIT_LABELS

GAP_LABEL_TO_ID: Mapping[str, int] = MappingProxyType(
    {label: index for index, label in enumerate(GAP_PUNCTUATION_LABELS)}
)
GAP_ID_TO_LABEL: tuple[str, ...] = GAP_PUNCTUATION_LABELS

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
    "gap_id_to_label",
    "gap_label_to_id",
    "rule_id_to_label",
    "rule_tag_to_id",
    "token_id_to_label",
    "token_label_to_id",
]

