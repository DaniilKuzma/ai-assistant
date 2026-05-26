from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from src.grammar_gen.ast import (
    Clause,
    ComplexSentence,
    DashSubjectPredicateSentence,
    IntroductorySentence,
    NounPhrase,
    SimpleSentence,
    VerbPhrase,
)
from src.schema import WordToken

if TYPE_CHECKING:
    from src.grammar_gen.builders import GrammarBuilder
    from src.grammar_gen.randomness import RandomSource
    from src.grammar_gen.realizer import Realizer
    from src.grammar_gen.semantics import VerbFrame


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONSTRUCTIONS_DIR = PROJECT_ROOT / "lexicon" / "constructions"


@dataclass(frozen=True)
class ConstructionRole:
    name: str
    semantic_classes: tuple[str, ...]
    required: bool = True
    case: str = "nomn"
    allow_adjectives: bool = True


@dataclass(frozen=True)
class ConstructionPattern:
    id: str
    family: str
    surface: str
    roles: dict[str, ConstructionRole]
    allowed_rule_ids: tuple[str, ...]
    allowed_frame_ids: tuple[str, ...] = ()
    allowed_frame_families: tuple[str, ...] = ()
    punctuation_profile: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RenderedConstruction:
    pattern_id: str
    text: str
    ast: Any | None
    tokens: list[WordToken]
    safety_clauses: list[dict]
    metadata: dict[str, Any]


class GeneratedTextValidator:
    def validate(self, example: Any) -> list[str]:
        raise NotImplementedError


class SurfaceValidator(GeneratedTextValidator):
    def validate(self, example: Any) -> list[str]:
        from src.grammar_gen.safety import validate_surface

        reasons: list[str] = []
        for field_name in ("source_text", "target_text"):
            text = str(getattr(example, field_name, ""))
            reasons.extend(f"{field_name}:{reason}" for reason in validate_surface(text))
        return _dedupe(reasons)


class SemanticFrameValidator(GeneratedTextValidator):
    def validate(self, example: Any) -> list[str]:
        from src.grammar_gen.safety import _validate_safety_clause  # type: ignore[attr-defined]

        raw_clauses = getattr(example, "metadata", {}).get("safety_clauses", [])
        if not isinstance(raw_clauses, list):
            return ["invalid_safety_clauses"]
        reasons: list[str] = []
        for clause in raw_clauses:
            reasons.extend(_validate_safety_clause(clause))
        return _dedupe(reasons)


class BadPatternValidator(GeneratedTextValidator):
    def validate(self, example: Any) -> list[str]:
        from src.grammar_gen.safety import check_known_bad_phrases

        reasons: list[str] = []
        for field_name in ("source_text", "target_text"):
            text = str(getattr(example, field_name, ""))
            reasons.extend(f"{field_name}:{reason}" for reason in check_known_bad_phrases(text))
        return _dedupe(reasons)


class ConstructionBank:
    def __init__(self, patterns: tuple[ConstructionPattern, ...]) -> None:
        self._patterns = patterns
        self._by_id = {pattern.id: pattern for pattern in patterns}
        if len(self._by_id) != len(patterns):
            raise ValueError("Construction pattern ids must be unique.")
        for pattern in patterns:
            _validate_pattern(pattern)

    @property
    def patterns(self) -> tuple[ConstructionPattern, ...]:
        return self._patterns

    @classmethod
    def default(cls) -> "ConstructionBank":
        return _default_construction_bank()

    @classmethod
    def from_dir(cls, path: str | Path) -> "ConstructionBank":
        base_path = Path(path)
        if not base_path.exists():
            raise FileNotFoundError(f"Construction directory does not exist: {base_path}")

        patterns: list[ConstructionPattern] = []
        for yaml_path in sorted(base_path.glob("*.yaml")):
            with yaml_path.open("r", encoding="utf-8") as handle:
                raw = yaml.safe_load(handle) or {}
            raw_patterns = raw.get("patterns") if isinstance(raw, dict) else None
            if not isinstance(raw_patterns, list):
                raise ValueError(f"Construction file must contain a patterns list: {yaml_path}")
            patterns.extend(_pattern_from_mapping(item, yaml_path) for item in raw_patterns)

        if not patterns:
            raise ValueError(f"No construction patterns found in {base_path}")
        return cls(tuple(patterns))

    def patterns_for_rule(self, rule_id: str) -> tuple[ConstructionPattern, ...]:
        return tuple(
            pattern
            for pattern in self._patterns
            if rule_id in pattern.allowed_rule_ids and not pattern.metadata.get("context_only")
        )

    def pattern(self, pattern_id: str) -> ConstructionPattern:
        try:
            return self._by_id[pattern_id]
        except KeyError as exc:
            raise ValueError(f"Unknown construction pattern: {pattern_id!r}.") from exc

    def sample_for_rule(
        self,
        rule_id: str,
        rng: RandomSource,
        family: str | None = None,
        *,
        include_context: bool = False,
    ) -> ConstructionPattern:
        candidates = tuple(
            pattern
            for pattern in self._patterns
            if rule_id in pattern.allowed_rule_ids
            and (family is None or pattern.family == family)
            and (include_context or not pattern.metadata.get("context_only"))
        )
        if not candidates:
            family_detail = f" and family {family!r}" if family else ""
            raise ValueError(f"No construction patterns for rule {rule_id!r}{family_detail}.")
        return rng.choice(candidates)

    def render(
        self,
        pattern: ConstructionPattern,
        builder: GrammarBuilder,
        realizer: Realizer,
        rng: RandomSource,
    ) -> RenderedConstruction:
        with builder.construction_context(pattern.id):
            render_type = str(pattern.metadata.get("render_type") or pattern.family)
            if render_type == "simple_transitive":
                return _render_simple_transitive(pattern, builder, realizer, rng)
            if render_type == "subordinate_that":
                return _render_subordinate_that(pattern, builder, realizer, rng)
            if render_type == "fixed_text":
                return _render_fixed_text(pattern, realizer, rng)
            if render_type == "dash_nominal":
                return _render_dash_nominal(pattern, builder, realizer, rng)
            if render_type == "homogeneous_objects":
                return _render_homogeneous_objects(pattern, builder, realizer, rng)
            if render_type == "adversative":
                return _render_adversative(pattern, builder, realizer, rng)
            if render_type == "introductory":
                return _render_introductory(pattern, builder, realizer, rng)
            if render_type == "tsya_ttsya":
                return _render_tsya_ttsya(pattern, builder, realizer, rng)
            if render_type == "hyphen_particle":
                return _render_hyphen_particle(pattern, builder, realizer, rng)
            if render_type == "hyphen_koe":
                return _render_hyphen_koe(pattern, builder, realizer, rng)
            if render_type == "po_adverb":
                return _render_po_adverb(pattern, builder, realizer, rng)
            if render_type == "takzhe_comparison":
                return _render_takzhe_comparison(pattern, builder, realizer, rng)
            if render_type == "tozhe_same":
                return _render_tozhe_same(pattern, builder, realizer, rng)
            if render_type == "zato_preposition":
                return _render_zato_preposition(pattern, builder, realizer, rng)
            if render_type == "hyphen_koe_negative":
                return _render_hyphen_koe_negative(pattern, builder, realizer, rng)
            if render_type == "po_adverb_negative":
                return _render_po_adverb_negative(pattern, builder, realizer, rng)
        raise ValueError(f"Unsupported construction render type: {render_type!r}.")


def _render_simple_transitive(
    pattern: ConstructionPattern,
    builder: GrammarBuilder,
    realizer: Realizer,
    rng: RandomSource,
) -> RenderedConstruction:
    frame = _choose_frame(pattern, builder, rng)
    clause = builder._clause_for_frame(
        frame,
        allow_adverbs=False,
        allow_adverbials=bool(pattern.metadata.get("allow_adverbials", False)),
    )
    adverbs = tuple(_string_list(pattern.metadata.get("adverbs")))
    if not adverbs and pattern.metadata.get("adverb"):
        adverbs = (str(pattern.metadata["adverb"]),)
    if adverbs or pattern.metadata.get("negated"):
        clause = replace(
            clause,
            predicate=replace(
                clause.predicate,
                adverbs=adverbs,
                negated=bool(pattern.metadata.get("negated", clause.predicate.negated)),
            ),
        )
    sentence = SimpleSentence(clause=clause, final_punctuation=".")
    text = realizer.render_sentence(sentence)
    return _rendered(
        pattern,
        text,
        sentence,
        realizer,
        safety_clauses=_safety_for_ast(sentence, builder),
        extra_metadata={"frame_id": frame.frame_id},
    )


def _render_subordinate_that(
    pattern: ConstructionPattern,
    builder: GrammarBuilder,
    realizer: Realizer,
    rng: RandomSource,
) -> RenderedConstruction:
    main_frame = _choose_frame(pattern, builder, rng)
    subordinate_frame_ids = _string_list(pattern.metadata.get("subordinate_frame_ids"))
    subordinate_frame = builder._frame_by_id(rng.choice(tuple(subordinate_frame_ids)))
    tense = "present" if subordinate_frame.frame_family == "content" else "past"
    sentence = ComplexSentence(
        main=builder._clause_for_frame(main_frame, allow_adverbs=False, allow_adverbials=False),
        conjunction="что",
        subordinate=builder._clause_for_frame(
            subordinate_frame,
            tense=tense,
            allow_adverbs=False,
            allow_adverbials=False,
        ),
        comma_before_conjunction=True,
        final_punctuation=".",
    )
    text = realizer.render_sentence(sentence)
    return _rendered(
        pattern,
        text,
        sentence,
        realizer,
        safety_clauses=_safety_for_ast(sentence, builder),
        extra_metadata={
            "main_frame_id": main_frame.frame_id,
            "subordinate_frame_id": subordinate_frame.frame_id,
        },
    )


def _render_fixed_text(
    pattern: ConstructionPattern,
    realizer: Realizer,
    rng: RandomSource,
) -> RenderedConstruction:
    texts = tuple(_string_list(pattern.metadata.get("texts")) or [pattern.surface])
    text = rng.choice(texts)
    return _rendered(pattern, text, None, realizer, safety_clauses=[])


def _render_dash_nominal(
    pattern: ConstructionPattern,
    builder: GrammarBuilder,
    realizer: Realizer,
    rng: RandomSource,
) -> RenderedConstruction:
    pairs = pattern.metadata.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError(f"Dash nominal pattern {pattern.id!r} must define pairs.")
    pair = rng.choice(tuple(pairs))
    if not isinstance(pair, dict):
        raise ValueError(f"Invalid dash nominal pair in {pattern.id!r}.")
    subject = builder._noun_phrase(builder._noun_by_lemma(str(pair["subject"])), allow_adjectives=False)
    predicate = builder._noun_phrase(builder._noun_by_lemma(str(pair["predicate"])), allow_adjectives=False)
    sentence = DashSubjectPredicateSentence(
        subject=subject,
        predicate_nominal=predicate,
        pair_id=str(pair["id"]),
        final_punctuation=".",
    )
    text = realizer.render_sentence(sentence).replace(" - ", " \u2014 ")
    return _rendered(
        pattern,
        text,
        sentence,
        realizer,
        safety_clauses=[],
        extra_metadata={"dash_pair_id": sentence.pair_id},
    )


def _render_homogeneous_objects(
    pattern: ConstructionPattern,
    builder: GrammarBuilder,
    realizer: Realizer,
    rng: RandomSource,
) -> RenderedConstruction:
    frame = _choose_frame(pattern, builder, rng)
    subject = _role_np(pattern, "subject", builder, rng)
    verb = realizer.morphology.inflect_verb_past(frame.verb_lemma, subject.gender, subject.number)
    objects = _distinct_objects_for_frame(builder, frame, rng, count=3)
    rendered_objects = [realizer.render_np(obj) for obj in objects]
    text = (
        f"{realizer.render_np(subject)} {verb} "
        f"{rendered_objects[0]}, {rendered_objects[1]} и {rendered_objects[2]}."
    )
    sentence = SimpleSentence(
        Clause(
            subject=subject,
            predicate=VerbPhrase(
                verb_lemma=frame.verb_lemma,
                object_np=objects[0],
                frame_id=frame.frame_id,
            ),
        )
    )
    return _rendered(
        pattern,
        _capitalize_first(text),
        sentence,
        realizer,
        safety_clauses=_safety_for_ast(sentence, builder),
        extra_metadata={
            "frame_id": frame.frame_id,
            "homogeneous_marker": rendered_objects[1],
            "homogeneous_object_lemmas": [obj.noun_lemma for obj in objects],
        },
    )


def _render_adversative(
    pattern: ConstructionPattern,
    builder: GrammarBuilder,
    realizer: Realizer,
    rng: RandomSource,
) -> RenderedConstruction:
    main_frame_ids = _string_list(pattern.metadata.get("main_frame_ids"))
    second_frame_ids = _string_list(pattern.metadata.get("second_frame_ids"))
    main_frame = builder._frame_by_id(rng.choice(tuple(main_frame_ids)))
    second_frame = builder._frame_by_id(rng.choice(tuple(second_frame_ids)))
    conjunction = str(pattern.metadata.get("conjunction") or rng.choice(("но", "а")))
    second = builder._clause_for_frame(second_frame, allow_adverbs=False, allow_adverbials=False)
    if pattern.metadata.get("negated_second", True):
        second = replace(second, predicate=replace(second.predicate, negated=True))
    sentence = ComplexSentence(
        main=builder._clause_for_frame(main_frame, allow_adverbs=False, allow_adverbials=False),
        conjunction=conjunction,
        subordinate=second,
        comma_before_conjunction=True,
        final_punctuation=".",
    )
    text = realizer.render_sentence(sentence)
    marker = f"{conjunction} {realizer.render_np(second.subject)}"
    return _rendered(
        pattern,
        text,
        sentence,
        realizer,
        safety_clauses=_safety_for_ast(sentence, builder),
        extra_metadata={
            "main_frame_id": main_frame.frame_id,
            "second_frame_id": second_frame.frame_id,
            "adversative_marker": marker,
        },
    )


def _render_introductory(
    pattern: ConstructionPattern,
    builder: GrammarBuilder,
    realizer: Realizer,
    rng: RandomSource,
) -> RenderedConstruction:
    frame = _choose_frame(pattern, builder, rng)
    positions = tuple(_string_list(pattern.metadata.get("positions")) or ["initial"])
    words = tuple(_string_list(pattern.metadata.get("introductory_words")) or ["конечно"])
    sentence = IntroductorySentence(
        introductory=rng.choice(words),
        clause=builder._clause_for_frame(frame, allow_adverbs=False, allow_adverbials=False),
        position=rng.choice(positions),
        final_punctuation=".",
    )
    text = realizer.render_sentence(sentence)
    return _rendered(
        pattern,
        text,
        sentence,
        realizer,
        safety_clauses=_safety_for_ast(sentence, builder),
        extra_metadata={"introductory_position": sentence.position, "frame_id": frame.frame_id},
    )


def _render_tsya_ttsya(
    pattern: ConstructionPattern,
    builder: GrammarBuilder,
    realizer: Realizer,
    rng: RandomSource,
) -> RenderedConstruction:
    verbs = pattern.metadata.get("verbs")
    if not isinstance(verbs, list) or not verbs:
        raise ValueError(f"tsya_ttsya pattern {pattern.id!r} must define verbs.")
    pair = rng.choice(tuple(verbs))
    infinitive = str(pair["infinitive"])
    finite = str(pair["finite"])
    subject = _role_np(pattern, "subject", builder, rng)
    subject_surface = _capitalize_first(realizer.render_np(subject))
    direction = rng.choice(("infinitive", "finite"))
    if direction == "infinitive":
        want = realizer.morphology.inflect_verb_past("хотеть", subject.gender, subject.number)
        text = f"{subject_surface} {want} {infinitive}."
    else:
        text = f"{subject_surface} {finite} утром."
    return _rendered(
        pattern,
        text,
        None,
        realizer,
        safety_clauses=[],
        extra_metadata={
            "tsya_direction": direction,
            "tsya_infinitive": infinitive,
            "tsya_finite": finite,
        },
    )


def _render_hyphen_particle(
    pattern: ConstructionPattern,
    builder: GrammarBuilder,
    realizer: Realizer,
    rng: RandomSource,
) -> RenderedConstruction:
    cases = pattern.metadata.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"hyphen particle pattern {pattern.id!r} must define cases.")
    case = rng.choice(tuple(cases))
    target_subject = str(case["target_subject"])
    source_subject = str(case["source_subject"])
    predicate = _pronoun_predicate(pattern, builder, realizer, rng)
    target = f"{target_subject} {predicate}"
    source = f"{source_subject} {predicate}"
    return _rendered(
        pattern,
        target,
        None,
        realizer,
        safety_clauses=[],
        extra_metadata={
            "source_text": source,
            "token_sequence": list(case["token_sequence"]),
            "token_label": str(case["token_label"]),
        },
    )


def _render_hyphen_koe(
    pattern: ConstructionPattern,
    builder: GrammarBuilder,
    realizer: Realizer,
    rng: RandomSource,
) -> RenderedConstruction:
    predicate = _pronoun_predicate(pattern, builder, realizer, rng)
    target = f"Кое-кто {predicate}"
    source = f"Кое кто {predicate}"
    return _rendered(
        pattern,
        target,
        None,
        realizer,
        safety_clauses=[],
        extra_metadata={
            "source_text": source,
            "token_sequence": ["Кое", "кто"],
            "token_label": "HYPHENATE_KOE",
        },
    )


def _render_po_adverb(
    pattern: ConstructionPattern,
    builder: GrammarBuilder,
    realizer: Realizer,
    rng: RandomSource,
) -> RenderedConstruction:
    subject = _role_np(pattern, "subject", builder, rng)
    verb = realizer.morphology.inflect_verb_past(
        rng.choice(tuple(_string_list(pattern.metadata.get("verb_lemmas")))),
        subject.gender,
        subject.number,
    )
    text = _capitalize_first(f"{realizer.render_np(subject)} {verb} по-русски.")
    return _rendered(
        pattern,
        text,
        None,
        realizer,
        safety_clauses=[],
        extra_metadata={
            "source_text": text.replace("по-русски", "по русски"),
            "token_sequence": ["по", "русски"],
            "token_label": "HYPHENATE_PO_ADVERB",
        },
    )


def _render_takzhe_comparison(
    pattern: ConstructionPattern,
    builder: GrammarBuilder,
    realizer: Realizer,
    rng: RandomSource,
) -> RenderedConstruction:
    subject = _role_np(pattern, "subject", builder, rng)
    verb = realizer.morphology.inflect_verb_past("сделать", subject.gender, subject.number)
    text = _capitalize_first(f"{realizer.render_np(subject)} {verb} так же, как эксперт.")
    return _rendered(pattern, text, None, realizer, safety_clauses=[])


def _render_tozhe_same(
    pattern: ConstructionPattern,
    builder: GrammarBuilder,
    realizer: Realizer,
    rng: RandomSource,
) -> RenderedConstruction:
    subject = _role_np(pattern, "subject", builder, rng)
    if rng.chance(0.5):
        verb = realizer.morphology.inflect_verb_past("выбрать", subject.gender, subject.number)
        text = _capitalize_first(f"{realizer.render_np(subject)} {verb} то же самое.")
    else:
        verb = realizer.morphology.inflect_verb_past("сделать", subject.gender, subject.number)
        text = _capitalize_first(f"{realizer.render_np(subject)} {verb} то же, что эксперт.")
    return _rendered(pattern, text, None, realizer, safety_clauses=[])


def _render_zato_preposition(
    pattern: ConstructionPattern,
    builder: GrammarBuilder,
    realizer: Realizer,
    rng: RandomSource,
) -> RenderedConstruction:
    subject = _role_np(pattern, "subject", builder, rng)
    verb = realizer.morphology.inflect_verb_past("голосовать", subject.gender, subject.number)
    text = _capitalize_first(f"{realizer.render_np(subject)} {verb} за то решение.")
    return _rendered(pattern, text, None, realizer, safety_clauses=[])


def _render_hyphen_koe_negative(
    pattern: ConstructionPattern,
    builder: GrammarBuilder,
    realizer: Realizer,
    rng: RandomSource,
) -> RenderedConstruction:
    subject = _role_np(pattern, "subject", builder, rng)
    verb = realizer.morphology.inflect_verb_past("спросить", subject.gender, subject.number)
    text = _capitalize_first(f"{realizer.render_np(subject)} кое у кого {verb}.")
    return _rendered(pattern, text, None, realizer, safety_clauses=[])


def _render_po_adverb_negative(
    pattern: ConstructionPattern,
    builder: GrammarBuilder,
    realizer: Realizer,
    rng: RandomSource,
) -> RenderedConstruction:
    subject = _role_np(pattern, "subject", builder, rng)
    verb = realizer.morphology.inflect_verb_past("идти", subject.gender, subject.number)
    text = _capitalize_first(f"{realizer.render_np(subject)} {verb} по русской дороге.")
    return _rendered(pattern, text, None, realizer, safety_clauses=[])


def _pronoun_predicate(
    pattern: ConstructionPattern,
    builder: GrammarBuilder,
    realizer: Realizer,
    rng: RandomSource,
) -> str:
    frame_ids = _string_list(pattern.metadata.get("predicate_frame_ids"))
    frame = builder._frame_by_id(rng.choice(tuple(frame_ids)))
    subject = builder._noun_phrase(builder._noun_by_lemma("студент"), allow_adjectives=False)
    predicate = VerbPhrase(
        verb_lemma=frame.verb_lemma,
        object_np=builder._object_for_frame(frame),
        frame_id=frame.frame_id,
    )
    return f"{realizer.render_vp(predicate, subject)}."


def _rendered(
    pattern: ConstructionPattern,
    text: str,
    ast: Any | None,
    realizer: Realizer,
    *,
    safety_clauses: list[dict],
    extra_metadata: dict[str, Any] | None = None,
) -> RenderedConstruction:
    metadata = dict(pattern.metadata)
    metadata.pop("pairs", None)
    metadata.pop("texts", None)
    metadata.pop("verbs", None)
    metadata.pop("cases", None)
    metadata.update(extra_metadata or {})
    metadata.update(
        {
            "uses_construction_bank": True,
            "construction_id": pattern.id,
            "construction_family": pattern.family,
            "safety_clauses": safety_clauses,
        }
    )
    if "uses_safety_clauses" not in metadata:
        metadata["uses_safety_clauses"] = bool(safety_clauses)
    return RenderedConstruction(
        pattern_id=pattern.id,
        text=text,
        ast=ast,
        tokens=realizer.tokenize_words_with_offsets(text),
        safety_clauses=safety_clauses,
        metadata=metadata,
    )


def _choose_frame(pattern: ConstructionPattern, builder: GrammarBuilder, rng: RandomSource) -> VerbFrame:
    frame_ids = pattern.allowed_frame_ids or tuple(_string_list(pattern.metadata.get("frame_ids")))
    if not frame_ids:
        raise ValueError(f"Construction pattern {pattern.id!r} must define frame ids.")
    return builder._frame_by_id(rng.choice(tuple(frame_ids)))


def _role_np(
    pattern: ConstructionPattern,
    role_name: str,
    builder: GrammarBuilder,
    rng: RandomSource,
) -> NounPhrase:
    role = pattern.roles[role_name]
    entry = builder.lexicon.random_noun_for_classes(role.semantic_classes, rng)
    return builder._noun_phrase(
        entry,
        case=role.case,
        allow_adjectives=role.allow_adjectives,
    )


def _distinct_objects_for_frame(
    builder: GrammarBuilder,
    frame: VerbFrame,
    rng: RandomSource,
    *,
    count: int,
) -> tuple[NounPhrase, ...]:
    objects: list[NounPhrase] = []
    used: set[str] = set()
    attempts = 0
    while len(objects) < count and attempts < 100:
        attempts += 1
        candidate = builder._object_for_frame(frame)
        if candidate is None or candidate.noun_lemma in used:
            continue
        used.add(candidate.noun_lemma)
        objects.append(candidate)
    if len(objects) != count:
        raise ValueError(f"Could not fill {count} distinct objects for frame {frame.frame_id!r}.")
    return tuple(objects)


def _safety_for_ast(ast: Any, builder: GrammarBuilder) -> list[dict]:
    from src.grammar_gen.safety import safety_clauses_for_ast

    return safety_clauses_for_ast(ast, builder.lexicon)


def _pattern_from_mapping(data: Any, path: Path) -> ConstructionPattern:
    if not isinstance(data, dict):
        raise ValueError(f"Invalid construction pattern in {path}: expected mapping.")
    roles = data.get("roles")
    if not isinstance(roles, dict) or not roles:
        raise ValueError(f"Construction pattern {data.get('id')!r} in {path} must define roles.")
    return ConstructionPattern(
        id=str(data["id"]).strip(),
        family=str(data["family"]).strip(),
        surface=str(data["surface"]).strip(),
        roles={name: _role_from_mapping(name, role) for name, role in roles.items()},
        allowed_rule_ids=tuple(_required_string_list(data, "allowed_rule_ids")),
        allowed_frame_ids=tuple(_string_list(data.get("allowed_frame_ids"))),
        allowed_frame_families=tuple(_string_list(data.get("allowed_frame_families"))),
        punctuation_profile=dict(data.get("punctuation_profile") or {}),
        metadata=dict(data.get("metadata") or {}),
    )


def _role_from_mapping(name: str, data: Any) -> ConstructionRole:
    if not isinstance(data, dict):
        raise ValueError(f"Invalid role {name!r}: expected mapping.")
    classes = tuple(_required_string_list(data, "semantic_classes"))
    return ConstructionRole(
        name=name,
        semantic_classes=classes,
        required=bool(data.get("required", True)),
        case=str(data.get("case", "nomn") or "nomn"),
        allow_adjectives=bool(data.get("allow_adjectives", True)),
    )


def _validate_pattern(pattern: ConstructionPattern) -> None:
    if not pattern.id:
        raise ValueError("Construction pattern id must not be empty.")
    if not pattern.family:
        raise ValueError(f"Construction pattern {pattern.id!r} has empty family.")
    if not pattern.surface:
        raise ValueError(f"Construction pattern {pattern.id!r} has empty surface.")
    if not pattern.roles:
        raise ValueError(f"Construction pattern {pattern.id!r} has no roles.")
    if not pattern.allowed_rule_ids:
        raise ValueError(f"Construction pattern {pattern.id!r} has no allowed_rule_ids.")
    for role_name, role in pattern.roles.items():
        if role.name != role_name:
            raise ValueError(f"Construction role key mismatch in pattern {pattern.id!r}.")
        if not role.semantic_classes:
            raise ValueError(f"Construction role {role_name!r} in {pattern.id!r} has no semantic classes.")


def _required_string_list(data: dict[str, Any], key: str) -> list[str]:
    values = _string_list(data.get(key))
    if not values:
        raise ValueError(f"Expected non-empty string list for {key!r}.")
    return values


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item).strip()]
    return []


def _capitalize_first(text: str) -> str:
    if not text:
        return text
    return f"{text[0].upper()}{text[1:]}"


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


__all__ = [
    "BadPatternValidator",
    "ConstructionBank",
    "ConstructionPattern",
    "ConstructionRole",
    "GeneratedTextValidator",
    "RenderedConstruction",
    "SemanticFrameValidator",
    "SurfaceValidator",
]


@lru_cache(maxsize=1)
def _default_construction_bank() -> ConstructionBank:
    return ConstructionBank.from_dir(DEFAULT_CONSTRUCTIONS_DIR)
