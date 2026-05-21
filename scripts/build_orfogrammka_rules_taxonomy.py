from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any
from urllib.parse import urljoin

import requests
import yaml
from bs4 import BeautifulSoup


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rules.registry import all_rules, rule_by_id  # noqa: E402


RULES_PATH = PROJECT_ROOT / "configs" / "rules.yaml"
REPORTS_DIR = PROJECT_ROOT / "reports" / "rules_taxonomy"
ORTHOGRAPHY_URL = "https://orfogrammka.ru/орфография/"
PUNCTUATION_URL = "https://orfogrammka.ru/пунктуация/"
NUMBER_RE = re.compile(r"^(\d+(?:\.\d+)*)(?:\s+|$)")
WS_RE = re.compile(r"\s+")

ALLOWED_STATUSES = (
    "implemented",
    "partial",
    "candidate_only",
    "model_required",
    "syntax_required",
    "dictionary_model_required",
    "ner_required",
    "planned",
    "metadata_only",
    "disabled",
)
RULE_ID_ALIASES = {"frequent_errors": "frequent_error_exact"}

MAPPING_TEXT = """
yo_e_candidate	orthography_1_1_1_3	medium	needs_manual_review	е/ё candidate exists but policy is disabled and taxonomy point is narrower than generic candidate.
pattern_чя_ча	orthography_1_1_2_1	high	mapped	Exact bounded candidate for а/у after hissing consonants.
pattern_щя_ща	orthography_1_1_2_1	high	mapped	Exact bounded candidate for а/у after hissing consonants.
pattern_чю_чу	orthography_1_1_2_1	high	mapped	Exact bounded candidate for а/у after hissing consonants.
pattern_щю_щу	orthography_1_1_2_1	high	mapped	Exact bounded candidate for а/у after hissing consonants.
pattern_жы_жи	orthography_1_1_2_2	high	mapped	Exact bounded candidate for и/ы after hissing consonants and ц.
pattern_шы_ши	orthography_1_1_2_2	high	mapped	Exact bounded candidate for и/ы after hissing consonants and ц.
pattern_цы_ци	orthography_1_1_2_2	medium	mapped	Bounded ци/цы candidate maps to the nearest и/ы taxonomy point.
cy_exception	orthography_1_1_2_2	medium	mapped	Bounded цы exception candidate maps to the nearest и/ы taxonomy point.
pattern_жо_же	orthography_1_1_2_3	medium	mapped	Bounded о/е after hissing candidate; taxonomy is broader.
pattern_шо_ше	orthography_1_1_2_3	medium	mapped	Bounded о/е after hissing candidate; taxonomy is broader.
pattern_чо_че	orthography_1_1_2_3	medium	mapped	Bounded о/е after hissing candidate; taxonomy is broader.
pattern_що_ще	orthography_1_1_2_3	medium	mapped	Bounded о/е after hissing candidate; taxonomy is broader.
soft_to_hard_sign	orthography_1_1_4_1	high	mapped	Executable candidate for replacing wrong soft sign with hard sign after prefix.
missing_hard_sign	orthography_1_1_4_1	high	mapped	Executable candidate for missing hard sign after prefix.
tsya_soft_insert	orthography_1_1_4_4	high	mapped	Bounded -тся/-ться candidate for grammatical soft sign forms.
tsya_soft_delete	orthography_1_1_4_4	high	mapped	Bounded -тся/-ться candidate for grammatical soft sign forms.
prefix_pre_pri	orthography_1_2_3_2	high	mapped	Dictionary-backed при-/пре- candidate.
sdelat_prefix	orthography_1_3_1_4	high	mapped	Executable candidate for prefix spelling on з/с.
prefix_z_to_s	orthography_1_3_1_4	high	mapped	Executable candidate for prefix spelling on з/с.
prefix_s_to_z	orthography_1_3_1_4	high	mapped	Executable candidate for prefix spelling on з/с.
n_nn_adjective	orthography_1_3_6_5	medium	needs_manual_review	Dictionary-backed Н/НН adjective candidate covers a bounded part of this point.
n_nn_participle	orthography_1_3_6_6	medium	needs_manual_review	Dictionary-backed Н/НН participle candidate covers a bounded part of this point.
n_nn_deverbal_adjective	orthography_1_3_6_7	medium	needs_manual_review	Н/НН deverbal adjective candidate maps to nearest participle/adjective point.
n_nn_short_form	orthography_1_3_6_8	medium	needs_manual_review	Н/НН short-form candidate maps to derivative-word taxonomy point.
pol_polu_compounds	orthography_3_1_1_2_9	high	mapped	Bounded candidate for пол-/полу- compounds.
hyphen_koe_koy	orthography_3_4_2_1	high	mapped	Exact candidate for кое-/кой- pronoun words.
hyphen_particles	orthography_3_4_2_2	high	mapped	Exact candidate for -либо, -нибудь, -то pronoun postfixes.
hyphen_po_adverbs	orthography_3_5_3_2	high	mapped	Exact bounded candidate for по- adverbs.
context_vsledstvie	orthography_3_6_1_1	medium	needs_manual_review	Context split/join candidate maps to derived preposition spelling.
context_nesmotrya	orthography_3_6_1_1	low	needs_manual_review	Context split/join candidate is broader than taxonomy leaf.
context_tak_zhe	orthography_3_6_1_3	medium	needs_manual_review	Context pair maps to service-word split/join rules.
context_to_zhe	orthography_3_6_1_3	medium	needs_manual_review	Context pair maps to service-word split/join rules.
context_chto_by	orthography_3_6_1_3	medium	needs_manual_review	Context pair maps to service-word split/join rules.
context_za_to	orthography_3_6_3_2	low	needs_manual_review	Context pair is related to separate service-word spelling but needs review.
ne_adjective	orthography_3_7_1_4_1	medium	needs_manual_review	Bounded не + adjective split/join candidate maps to closest joined spelling leaf.
ne_adverb	orthography_3_7_1_4_1	medium	needs_manual_review	Bounded не + adverb split/join candidate maps to closest joined spelling leaf.
ne_participle	orthography_3_7_2_11	medium	needs_manual_review	Bounded не + participle candidate maps to separate full participle spelling.
ne_verb	orthography_3_7_2_1	high	mapped	Bounded не + verb split candidate maps to exact verb rule.
ni_stable_expression	orthography_1_2_9_8	high	mapped	Bounded не/ни stable expression candidate maps to stable ни combinations.
hyphen_whitelist	orthography_3_11	low	needs_manual_review	Legacy hyphen whitelist has no one exact leaf; dictionary spelling is safest bucket.
capitalization_ner	orthography_4_1_1	medium	needs_manual_review	NER-backed capitalization maps to personal names but also covers other entity types.
capitalization_sentence_start	orthography_4_12	high	mapped	Bounded sentence-start capitalization candidate maps to capitals after punctuation.
abbreviation_case_protection	orthography_5_1_1	high	mapped	Case candidate for known initial abbreviations.
dictionary_fuzzy	orthography_8_1	medium	needs_manual_review	Generic lexicon fuzzy spelling candidate maps to dictionary-word errors.
frequent_error_exact	orthography_8_1	high	mapped	Exact frequent-error whitelist maps to dictionary-word errors.
keyboard_typo_candidate	orthography_7_3	high	mapped	Keyboard-neighbor typo candidate maps to incorrect-letter typos.
swapped_letters_candidate	orthography_7_1	high	mapped	Adjacent-letter swap candidate maps to letter operation typos.
missing_letter_candidate	orthography_7_1	high	mapped	Missing-letter candidate maps to letter operation typos.
extra_letter_candidate	orthography_7_1	high	mapped	Extra-letter candidate maps to letter operation typos.
double_consonant_candidate	orthography_1_3_6_9	medium	needs_manual_review	Generic double-consonant candidate maps to closest double-consonant point.
final_punctuation_default	punctuation_1_1	high	mapped	Bounded final full-stop candidate maps to sentence-final punctuation.
subject_predicate_dash	punctuation_2_1_2	high	mapped	Current candidate targets the это/вот subject-predicate dash case.
homogeneous_comma	punctuation_4_2	high	mapped	Bounded repeated-conjunction homogeneous comma candidate.
enumeration_colon	punctuation_4_7	high	mapped	Bounded generalizing-word colon candidate.
detached_adverbial_comma	punctuation_5_4_1	high	mapped	Bounded initial gerundial-turnover comma candidate.
comparative_turnover_comma	punctuation_5_8_1	high	mapped	Bounded comparative marker comma candidate.
introductory_comma	punctuation_6_1_1	high	mapped	Bounded introductory-word comma candidate.
address_comma	punctuation_6_2_1	high	mapped	Bounded address comma candidate.
comma_conjunction	punctuation_7_1_1	medium	needs_manual_review	Adversative conjunction comma maps to closest compound-sentence point.
semicolon	punctuation_7_1_2	high	mapped	Bounded semicolon candidate maps to semicolon in compound sentences.
comma_subordinate	punctuation_7_2_1	high	mapped	Bounded subordinate comma candidate.
direct_speech_colon	punctuation_8_1_1	medium	needs_manual_review	Bounded colon before quoted direct speech maps to author words before speech.
direct_speech_quotes	punctuation_8_1_1	medium	needs_manual_review	Bounded quote normalization maps to author words before speech.
direct_speech_dash	punctuation_8_1_2	medium	needs_manual_review	Bounded dash candidate maps to direct speech before author words.
bracket_pair_balance	punctuation_9_5	medium	needs_manual_review	One-sided bracket balance candidate maps to punctuation with brackets.
quote_open	punctuation_9_6	medium	needs_manual_review	Opening quote candidate maps to punctuation with quotes.
quote_close	punctuation_9_6	medium	needs_manual_review	Closing quote candidate maps to punctuation with quotes.
quote_pair_balance	punctuation_9_6	medium	needs_manual_review	One-sided quote balance candidate maps to punctuation with quotes.
explanation_colon	punctuation_10_1	low	needs_manual_review	Explanatory colon has no exact BSp leaf on page; mapped to complex mark interaction.
consequence_dash	punctuation_2_3	low	needs_manual_review	Consequence dash maps to broad connecting dash function.
asyndetic_dash	punctuation_2_3	low	needs_manual_review	Asyndetic dash maps to broad connecting dash function.
punctuation_delete_replace	punctuation_11_6	high	mapped	Bounded duplicate/noise cleanup maps to excessive punctuation.
""".strip()

STATUS_BY_KEY = {
    "orthography_1_1_1_3": "disabled",
    "orthography_1_1_2_1": "implemented",
    "orthography_1_1_2_2": "implemented",
    "orthography_1_1_2_3": "partial",
    "orthography_1_1_4_1": "partial",
    "orthography_1_1_4_4": "model_required",
    "orthography_1_2_3_2": "model_required",
    "orthography_1_3_1_4": "implemented",
    "orthography_1_3_6_5": "partial",
    "orthography_1_3_6_6": "partial",
    "orthography_1_3_6_7": "partial",
    "orthography_1_3_6_8": "partial",
    "orthography_3_4_2_1": "candidate_only",
    "orthography_3_4_2_2": "candidate_only",
    "orthography_3_5_3_2": "candidate_only",
    "orthography_3_11": "candidate_only",
    "orthography_4_1_1": "ner_required",
    "orthography_4_12": "candidate_only",
    "orthography_5_1_1": "candidate_only",
    "orthography_8_1": "partial",
    "punctuation_1_1": "implemented",
    "punctuation_11_6": "candidate_only",
}
TESTS_BY_KEY = {
    "orthography_1_1_2_1": [
        "tests/test_orthography_rule_generation.py::test_candidate_generator_emits_spelling_pattern_candidates_for_hundreds_of_forms",
        "tests/test_synthetic_generator.py::test_cha_shcha_rule_generates_corruption_and_candidate_with_same_rule_id",
    ],
    "orthography_1_1_2_2": ["tests/test_orthography_rule_generation.py::test_candidate_generator_emits_spelling_pattern_candidates_for_hundreds_of_forms"],
    "orthography_1_3_1_4": ["tests/test_orthography_rule_generation.py::test_candidate_generator_emits_spelling_pattern_candidates_for_hundreds_of_forms"],
    "punctuation_1_1": ["tests/test_punctuation_rules.py::test_final_dot_is_available_as_gap_candidate_without_repeated_marks"],
}
KEY_NOTES = {
    "orthography_1_1_1_3": "е/ё candidate exists in code but is disabled by default; no dataset activation without explicit opt-in policy.",
    "orthography_4_1_1": "NER-backed capitalization path exists but is not a complete proper-name implementation.",
}

TRANSLIT = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e", "ж": "zh", "з": "z",
    "и": "i", "й": "y", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o", "п": "p", "р": "r",
    "с": "s", "т": "t", "у": "u", "ф": "f", "х": "h", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "sch",
    "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rules-path", default=str(RULES_PATH))
    parser.add_argument("--reports-dir", default=str(REPORTS_DIR))
    args = parser.parse_args()
    rules_path = Path(args.rules_path)
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    extracted_at = datetime.now(timezone.utc).isoformat()
    try:
        extracted = {
            "orthography": extract_page("orthography", ORTHOGRAPHY_URL),
            "punctuation": extract_page("punctuation", PUNCTUATION_URL),
        }
    except Exception as exc:
        write_blocked_report(reports_dir, extracted_at, exc)
        return 2
    old = yaml.safe_load(rules_path.read_text(encoding="utf-8")) if rules_path.exists() else {}
    registry = {rule.spec.id: rule for rule in all_rules()}
    by_key = mappings_by_key()
    config: dict[str, Any] = {
        "schema_version": 3,
        "source": {
            "taxonomy_provider": "orfogrammka",
            "orthography_url": ORTHOGRAPHY_URL,
            "punctuation_url": PUNCTUATION_URL,
            "extracted_at": extracted_at,
            "note": "Full taxonomy mirror from Orfogrammka pages. Implementation statuses are project-specific.",
        },
    }
    if isinstance(old, dict) and old.get("synthetic_generation"):
        config["synthetic_generation"] = old["synthetic_generation"]
    old_entries = {
        section: old.get(section, {}) if isinstance(old.get(section), dict) else {}
        for section in ("orthography", "punctuation")
    }
    for section in ("orthography", "punctuation"):
        config[section] = {
            item["matrix_key"]: build_entry(
                item,
                by_key.get(item["matrix_key"], []),
                registry,
                old_entries.get(section, {}).get(item["matrix_key"], {}),
            )
            for item in extracted[section]["entries"]
        }
    rules_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False, width=120), encoding="utf-8")
    write_reports(config, extracted, registry, reports_dir, extracted_at)
    return 0


def mappings_by_key() -> dict[str, list[dict[str, str]]]:
    by_key: dict[str, list[dict[str, str]]] = defaultdict(list)
    for line in MAPPING_TEXT.splitlines():
        rule_id, key, confidence, action, reason = line.split("\t", 4)
        by_key[key].append({"rule_id": rule_id, "key": key, "confidence": confidence, "action": action, "reason": reason})
    return by_key


def all_mappings() -> dict[str, dict[str, str]]:
    return {row["rule_id"]: row for rows in mappings_by_key().values() for row in rows}


def extract_page(section: str, url: str) -> dict[str, Any]:
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    response.raise_for_status()
    response.encoding = "utf-8"
    soup = BeautifulSoup(response.text, "html.parser")
    root = max(
        [ul for ul in soup.find_all("ul") if re.search(r"\b\d+(?:\.\d+)+\b", ul.get_text(" ", strip=True))],
        key=lambda ul: len(ul.find_all("li")),
    )
    entries: list[dict[str, Any]] = []
    used: set[str] = set()
    path_to_key: dict[tuple[str, ...], str] = {}

    def walk(ul: Any, parent_path: tuple[str, ...] = ()) -> None:
        for li in ul.find_all("li", recursive=False):
            text = direct_text(li)
            if not text:
                continue
            match = NUMBER_RE.match(text)
            oid = match.group(1) if match else None
            title = text[match.end() :].strip() if match else text
            child = li.find("ul", recursive=False)
            current_path = (*parent_path, title)
            key = matrix_key(section, oid, current_path, used)
            if oid is None:
                path_to_key[current_path] = key
            anchor = li.find("a", recursive=False)
            entries.append(
                {
                    "matrix_key": key,
                    "source_section": section,
                    "orfogrammka_id": oid,
                    "title": title,
                    "normalized_title": normalize_title(title),
                    "entry_type": "group" if child else "leaf_rule",
                    "parent_key": path_to_key.get(parent_path),
                    "parent_path": list(parent_path),
                    "depth": len(parent_path) + 1,
                    "order": len(entries) + 1,
                    "source_url": urljoin(url, anchor["href"]) if anchor and anchor.has_attr("href") else url,
                    "has_children": bool(child),
                }
            )
            if child:
                walk(child, current_path)

    walk(root)
    return {
        "source_url": url,
        "entries": entries,
        "counts": {
            "total": len(entries),
            "numbered": sum(1 for item in entries if item["orfogrammka_id"]),
            "groups": sum(1 for item in entries if not item["orfogrammka_id"]),
            "leaf_rules": sum(1 for item in entries if item["entry_type"] == "leaf_rule"),
            "max_depth": max(item["depth"] for item in entries),
        },
    }


def direct_text(li: Any) -> str:
    parts = []
    for child in li.contents:
        if getattr(child, "name", None) == "ul":
            break
        parts.append(child.get_text(" ", strip=True) if getattr(child, "name", None) is not None else str(child).strip())
    return WS_RE.sub(" ", " ".join(part for part in parts if part).replace("\xa0", " ")).strip()


def matrix_key(section: str, oid: str | None, path: tuple[str, ...], used: set[str]) -> str:
    key = f"{section}_{oid.replace('.', '_')}" if oid else f"{section}_group_{slug('_'.join(path))}"
    if key in used:
        key = f"{key}_{hashlib.sha1(' > '.join(path).encode('utf-8')).hexdigest()[:8]}"
    used.add(key)
    return key


def slug(text: str) -> str:
    translit = "".join(TRANSLIT.get(ch.lower(), ch.lower()) for ch in text)
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", translit).strip("_")) or "untitled"


def normalize_title(title: str) -> str:
    return WS_RE.sub(" ", title).strip().lower()


def build_entry(
    raw: dict[str, Any],
    mappings: list[dict[str, str]],
    registry: dict[str, Any],
    old_entry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rule_ids = [row["rule_id"] for row in mappings if row["rule_id"] in registry]
    status = status_for(raw, rule_ids)
    executable = bool(rule_ids) and status not in {"planned", "metadata_only", "disabled"}
    if raw["matrix_key"] == "orthography_1_1_1_3":
        executable = False
    requires = sorted(set(default_requires(raw, status)) | set(registry_requires(rule_ids, registry)))
    eligible, reason = dataset_eligibility(status, executable, rule_ids, requires)
    impl = {
        "status": status,
        "executable": executable,
        "rule_ids": rule_ids,
        "aliases": [alias for alias, canonical in RULE_ID_ALIASES.items() if canonical in rule_ids],
        "requires": requires,
        "notes": KEY_NOTES.get(raw["matrix_key"], default_notes(raw, status, rule_ids)),
    }
    if raw["matrix_key"] in TESTS_BY_KEY:
        impl["tests"] = TESTS_BY_KEY[raw["matrix_key"]]
    old_dataset = old_entry.get("dataset", {}) if isinstance(old_entry, dict) else {}
    dataset = {
        "eligible_now": eligible,
        "reason": reason,
        "last_known_candidate_recall": None,
        "last_known_eval_count": None,
        **pre_dataset_defaults(status, executable, rule_ids, requires, eligible, reason),
    }
    for key in PRE_DATASET_DATASET_FIELDS:
        if key in old_dataset:
            dataset[key] = old_dataset[key]
    return {
        "source_section": raw["source_section"],
        "orfogrammka_id": raw["orfogrammka_id"],
        "title": raw["title"],
        "normalized_title": raw["normalized_title"],
        "entry_type": raw["entry_type"],
        "parent_key": raw["parent_key"],
        "parent_path": raw["parent_path"],
        "depth": raw["depth"],
        "order": raw["order"],
        "source_url": raw["source_url"],
        "implementation": impl,
        "dataset": dataset,
    }


def status_for(raw: dict[str, Any], rule_ids: list[str]) -> str:
    if raw["matrix_key"] in STATUS_BY_KEY:
        return STATUS_BY_KEY[raw["matrix_key"]]
    if raw["entry_type"] == "group":
        return "metadata_only"
    if rule_ids:
        return "model_required"
    text = " ".join([raw["title"], *raw["parent_path"]]).lower()
    if raw["source_section"] == "punctuation":
        return "model_required" if "нейропунктуация" in text else "syntax_required"
    if any(token in text for token in ("личные имена", "географ", "организац", "учрежден", "документов", "должностей", "наград", "инициала")):
        return "ner_required"
    if any(token in text for token in ("падеж", "причаст", "сказуем", "подлежащ", "противопостав", "местоимен", "нареч", "служебных слов", "отрицанием")):
        return "syntax_required"
    if any(token in text for token in ("словар", "термин", "омофон", "иностран", "аббревиат", "сокращ", "безудар", "суффикс", "корн", "согласн", "гласн")):
        return "dictionary_model_required"
    return "planned"


def registry_requires(rule_ids: list[str], registry: dict[str, Any]) -> list[str]:
    result: set[str] = set()
    for rule_id in rule_ids:
        result.update(str(item) for item in getattr(registry[rule_id].spec, "requires", ()))
    return sorted(result)


def default_requires(raw: dict[str, Any], status: str) -> list[str]:
    if status in {"metadata_only", "planned"}:
        return []
    if status == "disabled":
        return ["dictionary", "model"]
    if status == "ner_required":
        return ["ner", "syntax", "model"]
    if status == "syntax_required":
        return ["syntax", "model"]
    if status == "dictionary_model_required":
        return ["dictionary", "model"]
    if status == "model_required":
        return ["model"]
    return []


def default_notes(raw: dict[str, Any], status: str, rule_ids: list[str]) -> str:
    if status == "metadata_only":
        return "Group heading from Orfogrammka taxonomy; not an executable rule."
    if rule_ids:
        return "Mapped to existing bounded project rule_id(s); implementation coverage is project-specific."
    return "Present in Orfogrammka taxonomy; no executable project rule path is currently mapped."


def dataset_eligibility(status: str, executable: bool, rule_ids: list[str], requires: list[str]) -> tuple[bool, str]:
    if status == "metadata_only":
        return False, "metadata_only"
    if status == "disabled":
        return False, "disabled"
    if status == "planned":
        return False, "planned"
    if not executable or not rule_ids:
        if "ner" in requires:
            return False, "needs_NER"
        if "syntax" in requires:
            return False, "needs_syntax"
        if "dictionary" in requires:
            return False, "needs_dictionary"
        return False, "no_candidate_path"
    if "ner" in requires:
        return False, "needs_NER"
    if status == "model_required":
        return True, "eligible_but_needs_threshold"
    if status in {"partial", "candidate_only"}:
        return True, "eligible_but_needs_more_training"
    return True, "eligible_candidate_backed"


PRE_DATASET_DATASET_FIELDS = (
    "production_ready_now",
    "training_eligible_now",
    "training_eligibility_decision",
    "training_eligibility_reason",
    "current_candidate_path",
    "current_synthetic_support",
    "current_hard_negative_support",
    "current_validator_support",
    "current_candidate_recall",
    "current_gap_coverage",
    "risk_level",
    "needs_before_training",
)


def pre_dataset_defaults(
    status: str,
    executable: bool,
    rule_ids: list[str],
    requires: list[str],
    eligible: bool,
    reason: str,
) -> dict[str, Any]:
    if status == "metadata_only":
        decision = "BLOCK_METADATA_ONLY"
    elif status == "planned":
        decision = "BLOCK_PLANNED"
    elif status == "disabled":
        decision = "BLOCK_DISABLED"
    elif not executable or not rule_ids:
        if "ner" in requires:
            decision = "BLOCK_NEEDS_NER"
        elif "syntax" in requires:
            decision = "BLOCK_NEEDS_SYNTAX"
        elif "dictionary" in requires:
            decision = "BLOCK_NEEDS_DICTIONARY"
        else:
            decision = "BLOCK_NO_CANDIDATE"
    elif eligible and status == "model_required":
        decision = "INCLUDE_AFTER_THRESHOLD_CALIBRATION"
    elif eligible and status in {"partial", "candidate_only"}:
        decision = "INCLUDE_AFTER_TRAINING"
    elif eligible:
        decision = "INCLUDE_NOW"
    else:
        decision = "BLOCK_NO_CANDIDATE"
    return {
        "production_ready_now": False,
        "training_eligible_now": bool(eligible),
        "training_eligibility_decision": decision,
        "training_eligibility_reason": reason,
        "current_candidate_path": bool(executable and rule_ids),
        "current_synthetic_support": bool(eligible),
        "current_hard_negative_support": False,
        "current_validator_support": False,
        "current_candidate_recall": None,
        "current_gap_coverage": None,
        "risk_level": "high" if any(item in {"syntax", "ner"} for item in requires) else ("medium" if "dictionary" in requires else "low"),
        "needs_before_training": ["none"],
    }


def write_reports(config: dict[str, Any], extracted: dict[str, Any], registry: dict[str, Any], reports_dir: Path, extracted_at: str) -> None:
    snapshot = {
        "extraction_timestamp": extracted_at,
        "source_urls": {"orthography": ORTHOGRAPHY_URL, "punctuation": PUNCTUATION_URL},
        "counts": {section: data["counts"] for section, data in extracted.items()},
        "orthography": extracted["orthography"],
        "punctuation": extracted["punctuation"],
    }
    (reports_dir / "orfogrammka_extraction_snapshot.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    inventory = inventory_rows(config)
    write_csv(reports_dir / "rules_taxonomy_inventory.csv", inventory, [
        "matrix_key", "source_section", "orfogrammka_id", "title", "entry_type", "parent_path", "depth", "source_url",
        "status", "executable", "rule_ids", "requires", "eligible_now", "reason",
    ])
    write_csv(reports_dir / "missing_orfogrammka_entries_report.csv", [row for row in inventory if row["executable"] != "true"], list(inventory[0]))
    mapping = mapping_rows(config, registry)
    write_csv(reports_dir / "current_rule_id_mapping.csv", mapping, [
        "rule_id", "canonical_rule_id", "source_module", "mapped_matrix_key", "mapped_orfogrammka_id", "mapped_title",
        "mapping_confidence", "mapping_reason", "action",
    ])
    write_csv(reports_dir / "implementation_status_summary.csv", [{"status": s, "count": Counter(row["status"] for row in inventory).get(s, 0)} for s in ALLOWED_STATUSES], ["status", "count"])
    grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in inventory:
        grouped[(row["eligible_now"], row["reason"])].append(row["matrix_key"])
    write_csv(
        reports_dir / "dataset_eligibility_summary.csv",
        [{"eligible_now": k[0], "reason": k[1], "count": len(v), "matrix_keys": json.dumps(v, ensure_ascii=False)} for k, v in sorted(grouped.items())],
        ["eligible_now", "reason", "count", "matrix_keys"],
    )
    write_update_report(config, extracted, inventory, mapping, reports_dir / "rules_taxonomy_update_report.md")


def inventory_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for section in ("orthography", "punctuation"):
        for key, entry in config[section].items():
            impl = entry["implementation"]
            dataset = entry["dataset"]
            rows.append({
                "matrix_key": key,
                "source_section": section,
                "orfogrammka_id": entry["orfogrammka_id"] or "",
                "title": entry["title"],
                "entry_type": entry["entry_type"],
                "parent_path": json.dumps(entry["parent_path"], ensure_ascii=False),
                "depth": entry["depth"],
                "source_url": entry["source_url"],
                "status": impl["status"],
                "executable": str(bool(impl["executable"])).lower(),
                "rule_ids": json.dumps(impl["rule_ids"], ensure_ascii=False),
                "requires": json.dumps(impl["requires"], ensure_ascii=False),
                "eligible_now": str(bool(dataset["eligible_now"])).lower(),
                "reason": dataset["reason"],
            })
    return rows


def mapping_rows(config: dict[str, Any], registry: dict[str, Any]) -> list[dict[str, str]]:
    mapping = all_mappings()
    by_key = {key: entry for section in ("orthography", "punctuation") for key, entry in config[section].items()}
    rows = []
    for rule_id in sorted(registry):
        row = mapping.get(rule_id)
        if not row:
            rows.append(mapping_row(rule_id, rule_id, source_module(rule_id, registry), "", "", "", "unmapped", "Registry rule_id has no taxonomy mapping.", "missing_from_taxonomy"))
            continue
        entry = by_key.get(row["key"], {})
        rows.append(mapping_row(rule_id, rule_id, source_module(rule_id, registry), row["key"], str(entry.get("orfogrammka_id") or ""), str(entry.get("title") or ""), row["confidence"], row["reason"], row["action"]))
    for raw, canonical in RULE_ID_ALIASES.items():
        base = next((row for row in rows if row["rule_id"] == canonical), None)
        rows.append(mapping_row(raw, canonical, "src.rules.rule_ids", base["mapped_matrix_key"] if base else "", base["mapped_orfogrammka_id"] if base else "", base["mapped_title"] if base else "", "high", f"Legacy alias normalized to {canonical}.", "alias"))
    return rows


def mapping_row(rule_id: str, canonical: str, module: str, key: str, oid: str, title: str, confidence: str, reason: str, action: str) -> dict[str, str]:
    return {
        "rule_id": rule_id, "canonical_rule_id": canonical, "source_module": module, "mapped_matrix_key": key,
        "mapped_orfogrammka_id": oid, "mapped_title": title, "mapping_confidence": confidence,
        "mapping_reason": reason, "action": action,
    }


def source_module(rule_id: str, registry: dict[str, Any]) -> str:
    return "src.rules.punctuation" if getattr(registry[rule_id].spec, "scope", "") == "punctuation_gap" else "src.rules.orthography"


def write_update_report(config: dict[str, Any], extracted: dict[str, Any], inventory: list[dict[str, Any]], mapping: list[dict[str, str]], path: Path) -> None:
    status_counts = Counter(row["status"] for row in inventory)
    reason_counts = Counter(row["reason"] for row in inventory)
    manual = [row for row in mapping if row["action"] == "needs_manual_review"]
    missing = [row for row in inventory if row["executable"] != "true"]
    lines = [
        "# Rules Taxonomy Update Report",
        "",
        "Final verdict: RULES_TAXONOMY_COMPLETE",
        "",
        "## Sources",
        f"- Orthography: {ORTHOGRAPHY_URL}",
        f"- Punctuation: {PUNCTUATION_URL}",
        "- Extraction status: OK",
        "",
        "## Counts",
        f"- orthography entries: {len(config['orthography'])}",
        f"- punctuation entries: {len(config['punctuation'])}",
        f"- total entries: {len(inventory)}",
        f"- orthography extracted numbered: {extracted['orthography']['counts']['numbered']}",
        f"- punctuation extracted numbered: {extracted['punctuation']['counts']['numbered']}",
        "",
        "## Implementation Status",
        *[f"- {status}: {status_counts.get(status, 0)}" for status in ALLOWED_STATUSES],
        "",
        "## Dataset Eligibility",
        f"- eligible_now=true: {sum(1 for row in inventory if row['eligible_now'] == 'true')}",
        f"- eligible_now=false: {sum(1 for row in inventory if row['eligible_now'] == 'false')}",
        *[f"- {reason}: {count}" for reason, count in sorted(reason_counts.items())],
        "",
        "## Current Rule ID Mapping",
        f"- unmapped current rule_ids: {sum(1 for row in mapping if row['action'] == 'missing_from_taxonomy')}",
        f"- manual review mappings: {len(manual)}",
        "",
        "## Taxonomy Entries Without Implementation",
        f"- count: {len(missing)}",
        *[f"- {row['matrix_key']}: {row['title']}" for row in missing[:50]],
        "",
        "## Manual Review",
        *[f"- {row['rule_id']} -> {row['mapped_matrix_key']} ({row['mapping_confidence']}): {row['mapping_reason']}" for row in manual],
        "",
        "## Warnings",
        "- No dataset build, training, threshold calibration, checkpoint, or production threshold mutation was performed.",
        "",
        "RULES_TAXONOMY_COMPLETE",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def write_blocked_report(reports_dir: Path, extracted_at: str, exc: Exception) -> None:
    lines = [
        "# Rules Taxonomy Update Report", "", "Final verdict: BLOCKED_NO_SOURCE_ACCESS", "",
        f"- attempted_at: {extracted_at}", f"- orthography URL: {ORTHOGRAPHY_URL}", f"- punctuation URL: {PUNCTUATION_URL}",
        f"- reason: {type(exc).__name__}: {exc}", "", "BLOCKED_NO_SOURCE_ACCESS",
    ]
    (reports_dir / "rules_taxonomy_update_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
