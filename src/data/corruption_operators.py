from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import re
from pathlib import Path
from typing import Any, Iterable, Protocol

import pandas as pd

from src.candidates.candidate_generator import CandidateGenerator
from src.data.atomic_verifier import verify_atomic_positive
from src.data.dataset_quality import positive_target_quality_pass
from src.data.dataset_verifiers import (
    CONTEXT_PAIRS,
    PUNCTUATION_RULE_FAMILIES,
    TYPO_RULE_FAMILIES,
    semantic_alignment_for_rule,
)
from src.rules.registry import rule_by_id
from src.validation.diff_analyzer import DiffAnalyzer
from src.validation.edit_classifier import is_allowed_edit_type


@dataclass(frozen=True)
class Opportunity:
    start: int
    end: int
    text: str
    rule_id: str
    evidence: dict[str, Any] = field(default_factory=dict)
    syntax_family: str = ""
    confidence: float = 1.0
    source_type: str = "corpus"


@dataclass(frozen=True)
class CorruptionResult:
    source: str
    target: str
    rule_id: str
    edits: list[dict[str, Any]]
    error_bearing_span: tuple[int, int]
    error_form: str
    target_form: str
    generation_strategy: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VerificationResult:
    passed: bool
    reason: str
    actual_error_family: str
    expected_error_family: str
    candidate_present: bool
    candidate_rule_ids: list[str]
    target_quality_pass: bool
    semantic_alignment_pass: bool
    gold_edit_count: int = 0
    strict_validator_passed: bool = False
    matched_candidate: dict[str, Any] | None = None


@dataclass(frozen=True)
class HardNegativeResult:
    source: str
    target: str
    rule_id: str
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)


class SemanticVerifier(Protocol):
    def verify(self, rule_id: str, source: str, target: str) -> SemanticVerification:
        ...


class HardNegativeGenerator(Protocol):
    def generate_hard_negatives(self, clean_sentence: str, syntax_analysis: Any | None = None) -> list[HardNegativeResult]:
        ...


class CorruptionOperator:
    rule_id: str
    error_type: str
    requires: tuple[str, ...]

    def find_opportunities(self, clean_sentence: str, syntax_analysis: Any | None = None) -> list[Opportunity]:
        raise NotImplementedError

    def corrupt(self, clean_sentence: str, opportunity: Opportunity) -> CorruptionResult:
        raise NotImplementedError

    def verify(self, source: str, target: str, opportunity: Opportunity) -> VerificationResult:
        raise NotImplementedError

    def generate_hard_negatives(self, clean_sentence: str, syntax_analysis: Any | None = None) -> list[HardNegativeResult]:
        return []


class RuleOperatorRegistry:
    def __init__(self) -> None:
        self._operators: dict[str, CorruptionOperator] = {}

    def register(self, operator: CorruptionOperator) -> None:
        rule_id = str(operator.rule_id)
        if not rule_id:
            raise ValueError("operator rule_id must not be empty")
        if rule_id in self._operators:
            raise ValueError(f"duplicate operator for rule_id: {rule_id}")
        self._operators[rule_id] = operator

    def get(self, rule_id: str) -> CorruptionOperator | None:
        return self._operators.get(str(rule_id))

    def has_operator(self, rule_id: str) -> bool:
        return str(rule_id) in self._operators

    def rule_ids(self) -> list[str]:
        return sorted(self._operators)

    def rows(self) -> list[dict[str, Any]]:
        return [
            {
                "rule_id": rule_id,
                "operator": type(operator).__name__,
                "error_type": operator.error_type,
                "requires": ",".join(operator.requires),
            }
            for rule_id, operator in sorted(self._operators.items())
        ]


@dataclass(frozen=True)
class PunctuationCorruptionOperator(CorruptionOperator):
    rule_id: str
    punctuation: str
    error_type: str = "punctuation"
    requires: tuple[str, ...] = ("syntax",)

    def find_opportunities(self, clean_sentence: str, syntax_analysis: Any | None = None) -> list[Opportunity]:
        del syntax_analysis
        opportunities: list[Opportunity] = []
        for start, end in _punctuation_spans_for_rule(self.rule_id, clean_sentence):
            opportunities.append(
                Opportunity(
                    start=start,
                    end=end,
                    text=clean_sentence[start:end],
                    rule_id=self.rule_id,
                    evidence={"punctuation": self.punctuation},
                    syntax_family=PUNCTUATION_RULE_FAMILIES.get(self.rule_id, self.rule_id),
                    confidence=1.0,
                    source_type="corpus",
                )
            )
        return opportunities

    def corrupt(self, clean_sentence: str, opportunity: Opportunity) -> CorruptionResult:
        source = clean_sentence[: opportunity.start] + clean_sentence[opportunity.end :]
        edits = _edits_for_pair(source, clean_sentence, self.rule_id)
        return CorruptionResult(
            source=source,
            target=clean_sentence,
            rule_id=self.rule_id,
            edits=edits,
            error_bearing_span=(opportunity.start, opportunity.start),
            error_form="",
            target_form=opportunity.text,
            generation_strategy="corpus_opportunity",
            metadata={"operator": type(self).__name__, "opportunity": asdict(opportunity)},
        )

    def verify(self, source: str, target: str, opportunity: Opportunity) -> VerificationResult:
        return _verification(self.rule_id, source, target, opportunity)


@dataclass(frozen=True)
class ReplacementCorruptionOperator(CorruptionOperator):
    rule_id: str
    replacements: tuple[tuple[str, str], ...]
    error_type: str = "spelling"
    requires: tuple[str, ...] = ()
    expected_family: str = "orthography"

    def find_opportunities(self, clean_sentence: str, syntax_analysis: Any | None = None) -> list[Opportunity]:
        del syntax_analysis
        lower = clean_sentence.lower()
        result: list[Opportunity] = []
        for target_form, error_form in self.replacements:
            for match in re.finditer(r"\b" + re.escape(target_form) + r"\b", lower):
                result.append(
                    Opportunity(
                        start=match.start(),
                        end=match.end(),
                        text=clean_sentence[match.start() : match.end()],
                        rule_id=self.rule_id,
                        evidence={"target_form": target_form, "error_form": error_form},
                        syntax_family=self.expected_family,
                        confidence=1.0,
                        source_type="corpus",
                    )
                )
        return result

    def corrupt(self, clean_sentence: str, opportunity: Opportunity) -> CorruptionResult:
        error_form = str(opportunity.evidence.get("error_form") or "")
        if opportunity.text[:1].isupper():
            error_form = error_form[:1].upper() + error_form[1:]
        source = clean_sentence[: opportunity.start] + error_form + clean_sentence[opportunity.end :]
        edits = _edits_for_pair(source, clean_sentence, self.rule_id)
        return CorruptionResult(
            source=source,
            target=clean_sentence,
            rule_id=self.rule_id,
            edits=edits,
            error_bearing_span=(opportunity.start, opportunity.start + len(error_form)),
            error_form=error_form,
            target_form=opportunity.text,
            generation_strategy="corpus_opportunity",
            metadata={"operator": type(self).__name__, "opportunity": asdict(opportunity)},
        )

    def verify(self, source: str, target: str, opportunity: Opportunity) -> VerificationResult:
        return _verification(self.rule_id, source, target, opportunity)


@dataclass(frozen=True)
class RuleCorruptionBackedOperator(CorruptionOperator):
    rule_id: str
    error_type: str = "spelling"
    requires: tuple[str, ...] = ()

    def find_opportunities(self, clean_sentence: str, syntax_analysis: Any | None = None) -> list[Opportunity]:
        del syntax_analysis
        rule = rule_by_id(self.rule_id)
        generator = getattr(rule, "generate_corruptions", None)
        if generator is None:
            return []
        opportunities: list[Opportunity] = []
        for corruption in generator(clean_sentence):
            if getattr(corruption, "rule_id", self.rule_id) != self.rule_id:
                continue
            opportunities.append(
                Opportunity(
                    start=int(corruption.start),
                    end=int(corruption.end),
                    text=clean_sentence[int(corruption.start) : int(corruption.end)],
                    rule_id=self.rule_id,
                    evidence={"error_form": str(corruption.replacement)},
                    syntax_family=str(getattr(corruption, "group", "") or self.rule_id),
                    confidence=1.0,
                    source_type="corpus",
                )
            )
        return opportunities

    def corrupt(self, clean_sentence: str, opportunity: Opportunity) -> CorruptionResult:
        error_form = str(opportunity.evidence.get("error_form") or "")
        source = clean_sentence[: opportunity.start] + error_form + clean_sentence[opportunity.end :]
        edits = _edits_for_pair(source, clean_sentence, self.rule_id)
        return CorruptionResult(
            source=source,
            target=clean_sentence,
            rule_id=self.rule_id,
            edits=edits,
            error_bearing_span=(opportunity.start, opportunity.start + len(error_form)),
            error_form=error_form,
            target_form=opportunity.text,
            generation_strategy="corpus_opportunity",
            metadata={"operator": type(self).__name__, "opportunity": asdict(opportunity)},
        )

    def verify(self, source: str, target: str, opportunity: Opportunity) -> VerificationResult:
        return _verification(self.rule_id, source, target, opportunity)


def build_default_operator_registry() -> RuleOperatorRegistry:
    registry = RuleOperatorRegistry()
    for rule_id, punctuation in {
        "comma_subordinate": ",",
        "comma_conjunction": ",",
        "introductory_comma": ",",
        "address_comma": ",",
        "homogeneous_comma": ",",
        "detached_adverbial_comma": ",",
        "detached_participial_comma": ",",
        "apposition_comma": ",",
        "clarification_comma": ",",
        "comparative_turnover_comma": ",",
        "subject_predicate_dash": " — ",
        "asyndetic_dash": " — ",
        "consequence_dash": " — ",
        "explanation_colon": ":",
        "enumeration_colon": ":",
        "enumeration_dash": " — ",
        "semicolon": ";",
        "direct_speech_quotes": "«»",
        "direct_speech_colon": ":",
        "direct_speech_dash": " — ",
        "quote_pair_balance": "«»",
        "bracket_pair_balance": "()",
        "punctuation_delete_replace": ",",
        "final_punctuation_default": ".",
    }.items():
        registry.register(PunctuationCorruptionOperator(rule_id=rule_id, punctuation=punctuation))

    for rule_id, pairs, family in _replacement_operator_specs():
        registry.register(ReplacementCorruptionOperator(rule_id=rule_id, replacements=pairs, expected_family=family))

    for rule_id in _rule_corruption_backed_rule_ids():
        if not registry.has_operator(rule_id):
            registry.register(RuleCorruptionBackedOperator(rule_id=rule_id))
    return registry


def resolve_operator_training_targets(
    *,
    candidate_rule_ids: Iterable[str],
    registry: RuleOperatorRegistry,
    excluded_rule_ids: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    excluded_rule_ids = excluded_rule_ids or {}
    rows: list[dict[str, Any]] = []
    for rule_id in sorted({str(item) for item in candidate_rule_ids if str(item)}):
        operator = registry.get(rule_id)
        if rule_id in excluded_rule_ids:
            include = False
            reason = excluded_rule_ids[rule_id]
        elif operator is None:
            include = False
            reason = "BLOCK_NO_OPERATOR"
        else:
            include = True
            reason = "registered_operator"
        rows.append(
            {
                "rule_id": rule_id,
                "include": include,
                "reason": reason,
                "operator": type(operator).__name__ if operator is not None else "",
                "error_type": operator.error_type if operator is not None else "",
                "requires": ",".join(operator.requires) if operator is not None else "",
            }
        )
    return rows


def write_operator_registry_reports(rows: list[dict[str, Any]], *, registry: RuleOperatorRegistry, reports_dir: str | Path) -> None:
    reports = Path(reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    registry_frame = pd.DataFrame(registry.rows(), columns=["rule_id", "operator", "error_type", "requires"])
    registry_frame.to_csv(reports / "operator_registry_report.csv", index=False)
    targets = pd.DataFrame(rows, columns=["rule_id", "include", "reason", "operator", "error_type", "requires"])
    targets.to_csv(reports / "operator_coverage_by_rule.csv", index=False)
    targets[~targets["include"].astype(bool)].to_csv(reports / "rules_without_operator.csv", index=False)


def verify_pair_for_rule(rule_id: str, source: str, target: str, *, candidate_generator: CandidateGenerator | None = None) -> VerificationResult:
    opportunity = Opportunity(0, 0, "", rule_id, source_type="existing_pair")
    return _verification(rule_id, source, target, opportunity, candidate_generator=candidate_generator)


def _verification(
    rule_id: str,
    source: str,
    target: str,
    opportunity: Opportunity,
    *,
    candidate_generator: CandidateGenerator | None = None,
) -> VerificationResult:
    del opportunity
    semantic = semantic_alignment_for_rule(rule_id, source, target)
    target_quality = positive_target_quality_pass(target)
    if source == target:
        return VerificationResult(
            passed=False,
            reason="identity_pair",
            actual_error_family=semantic.actual_error_family,
            expected_error_family=semantic.expected_error_family,
            candidate_present=False,
            candidate_rule_ids=[],
            target_quality_pass=target_quality,
            semantic_alignment_pass=False,
            gold_edit_count=0,
        )

    generator = candidate_generator or _candidate_generator_for_rule(rule_id, target)
    atomic = verify_atomic_positive(source, target, rule_id, candidate_generator=generator)
    if atomic.gold_edit_count != 1:
        passed = False
        reason = "non_atomic_edit_count"
    elif not semantic.semantic_alignment_pass:
        passed = False
        reason = semantic.reason
    elif not target_quality:
        passed = False
        reason = "target_quality_failed"
    elif not atomic.candidate_present:
        passed = False
        reason = "candidate_missing"
    elif not atomic.strict_validator_passed:
        passed = False
        reason = "strict_validator_rejected"
    elif not atomic.passed:
        passed = False
        reason = atomic.reason
    else:
        passed = True
        reason = "ok"

    return VerificationResult(
        passed=passed,
        reason=reason,
        actual_error_family=semantic.actual_error_family,
        expected_error_family=semantic.expected_error_family,
        candidate_present=atomic.candidate_present,
        candidate_rule_ids=atomic.candidate_rule_ids,
        target_quality_pass=target_quality,
        semantic_alignment_pass=semantic.semantic_alignment_pass,
        gold_edit_count=atomic.gold_edit_count,
        strict_validator_passed=atomic.strict_validator_passed,
        matched_candidate=atomic.matched_candidate,
    )


def _candidate_present(
    rule_id: str,
    source: str,
    target: str,
    *,
    candidate_generator: CandidateGenerator | None = None,
) -> tuple[bool, list[str]]:
    generator = candidate_generator or _candidate_generator_for_rule(rule_id, target)
    atomic = verify_atomic_positive(source, target, rule_id, candidate_generator=generator, require_strict_validator=False)
    return atomic.candidate_present, atomic.candidate_rule_ids


def _candidate_generator_for_rule(rule_id: str, target: str) -> CandidateGenerator:
    lexicon = []
    if rule_id in TYPO_RULE_FAMILIES or rule_id == "dictionary_fuzzy":
        lexicon.extend(_words(target))
    lexicon.extend(["молоко", "корова", "собака", "библиотека", "грамматика", "территория", "комиссия"])
    return CandidateGenerator(dictionary_lexicon=tuple(dict.fromkeys(lexicon)), dictionary_limit=4, syntax_provider=lambda _text: ())


def _edits_for_pair(source: str, target: str, rule_id: str) -> list[dict[str, Any]]:
    edits: list[dict[str, Any]] = []
    for edit in DiffAnalyzer().analyze(source, target, candidates=[]):
        if not is_allowed_edit_type(edit.edit_type):
            continue
        edits.append(
            {
                "source": edit.source,
                "replacement": edit.replacement,
                "edit_type": edit.edit_type,
                "start": edit.start,
                "end": edit.end,
                "status": edit.status,
                "reason": edit.reason,
                "confidence": edit.confidence,
                "rule_id": rule_id,
            }
        )
    if not edits and source != target:
        edits.append({"source": "", "replacement": "", "edit_type": "punctuation_insert", "start": -1, "end": -1, "rule_id": rule_id})
    return edits


def _punctuation_spans_for_rule(rule_id: str, text: str) -> list[tuple[int, int]]:
    if rule_id.startswith("comma_") or rule_id.endswith("_comma"):
        return [(match.start(), match.end()) for match in re.finditer(",", text)]
    if rule_id in {"subject_predicate_dash", "asyndetic_dash", "consequence_dash", "enumeration_dash", "direct_speech_dash"}:
        return [(match.start(), match.end()) for match in re.finditer(r"\s—\s", text)]
    if rule_id in {"explanation_colon", "enumeration_colon", "direct_speech_colon"}:
        return [(match.start(), match.end()) for match in re.finditer(":", text)]
    if rule_id == "semicolon":
        return [(match.start(), match.end()) for match in re.finditer(";", text)]
    if rule_id in {"direct_speech_quotes", "quote_pair_balance"}:
        return [(match.start(), match.end()) for match in re.finditer("[«»\"]", text)]
    if rule_id == "bracket_pair_balance":
        return [(match.start(), match.end()) for match in re.finditer(r"[\(\)\[\]\{\}]", text)]
    if rule_id == "final_punctuation_default":
        stripped = text.rstrip()
        if stripped.endswith((".", "!", "?")):
            end = len(stripped)
            return [(end - 1, end)]
    return []


def _replacement_operator_specs() -> list[tuple[str, tuple[tuple[str, str], ...], str]]:
    return [
        ("missing_letter_candidate", (("молоко", "млоко"), ("корова", "корва"), ("собака", "сбака"), ("библиотека", "библотека")), "missing_letter"),
        ("extra_letter_candidate", (("молоко", "молокоо"), ("корова", "коорова"), ("собака", "собакаа"), ("библиотека", "библиотекаа")), "extra_letter"),
        ("swapped_letters_candidate", (("молоко", "молкоо"), ("корова", "коорва"), ("собака", "соабка"), ("библиотека", "бибилотека")), "swapped_letters"),
        ("keyboard_typo_candidate", (("молоко", "молокл"), ("корова", "клрова"), ("собака", "слбака"), ("грамматика", "грсмматика")), "keyboard_typo"),
        ("double_consonant_candidate", (("грамматика", "граматика"), ("территория", "територия"), ("комиссия", "комисия"), ("профессия", "проффесия")), "double_consonant"),
        ("dictionary_fuzzy", (("библиотека", "библеотека"), ("территория", "територия"), ("грамматика", "грамматикаа")), "dictionary_fuzzy"),
        ("context_tak_zhe", (("так же", "также"), ("также", "так же")), "context_pair"),
        ("context_to_zhe", (("то же", "тоже"), ("тоже", "то же")), "context_pair"),
        ("context_chto_by", (("что бы", "чтобы"), ("чтобы", "что бы")), "context_pair"),
        ("context_za_to", (("зато", "за то"), ("за то", "зато")), "context_pair"),
        ("context_vsledstvie", (("вследствие", "в следствие"), ("в следствие", "вследствие")), "context_pair"),
        ("context_nesmotrya", (("несмотря", "не смотря"), ("не смотря", "несмотря")), "context_pair"),
    ]


def _rule_corruption_backed_rule_ids() -> tuple[str, ...]:
    return (
        "frequent_error_exact",
        "pattern_жы_жи",
        "pattern_шы_ши",
        "pattern_чя_ча",
        "pattern_щя_ща",
        "pattern_чю_чу",
        "pattern_щю_щу",
        "pattern_цы_ци",
        "pattern_жо_же",
        "pattern_шо_ше",
        "pattern_чо_че",
        "pattern_що_ще",
        "cy_exception",
        "soft_to_hard_sign",
        "missing_hard_sign",
        "sdelat_prefix",
        "prefix_z_to_s",
        "prefix_s_to_z",
        "prefix_pre_pri",
        "ne_verb",
        "ne_adjective",
        "ne_participle",
        "ne_adverb",
        "ne_short_form",
        "ne_predicative",
        "ni_stable_expression",
        "ni_particle_context",
        "n_nn_adjective",
        "n_nn_participle",
        "n_nn_deverbal_adjective",
        "n_nn_short_form",
        "tsya_soft_delete",
        "tsya_soft_insert",
        "hyphen_whitelist",
        "hyphen_particles",
        "hyphen_koe_koy",
        "hyphen_po_adverbs",
        "pol_polu_compounds",
        "capitalization_sentence_start",
        "abbreviation_case_protection",
    )


def _words(text: str) -> list[str]:
    return re.findall(r"[а-яёА-ЯЁ-]+", text.lower())
