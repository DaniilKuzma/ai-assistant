from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from src.candidates.morphology import has_pos, is_known_word, normal_forms, parses
from src.preprocessing.tokenizer import tokenize_words
from src.rules.base import RuleCandidate, RuleContext, RuleCorruption, RuleEdit, RuleMode, RuleSpec


VERB_POSES = frozenset({"VERB", "INFN"})
TSYA_CONTEXT_POSES = frozenset({"VERB", "INFN"})
NE_FULL_ADJECTIVE_POSES = frozenset({"ADJF"})
NE_SHORT_FORM_POSES = frozenset({"ADJS"})
NE_PARTICIPLE_POSES = frozenset({"PRTF", "PRTS"})
NE_ADVERB_POSES = frozenset({"ADVB"})
NE_PREDICATIVE_POSES = frozenset({"PRED", "ADVB"})
N_NN_ADJECTIVE_POSES = frozenset({"ADJF", "ADJS"})
N_NN_PARTICIPLE_POSES = frozenset({"PRTF", "PRTS"})
NE_REQUIRES = ("morphology", "syntax", "model")
NI_REQUIRES = ("morphology", "syntax", "model")
N_NN_REQUIRES = ("morphology", "syntax", "dictionary", "model")
CONTEXT_REQUIRES = ("syntax", "morphology", "model")
ORTHOGRAPHY_DOMAIN = "orthography"

NI_STABLE_EXPRESSION_PAIRS = (
    ("не разу", "ни разу"),
    ("ни разу", "не разу"),
    ("не в коем случае", "ни в коем случае"),
    ("ни в коем случае", "не в коем случае"),
    ("во что бы то не стало", "во что бы то ни стало"),
    ("во что бы то ни стало", "во что бы то не стало"),
)
NE_JOINED_NORMAL_FORMS = frozenset(
    {
        "ненавидеть",
        "негодовать",
        "нездоровиться",
        "недоумевать",
        "недосмотреть",
        "недооценить",
        "недосыпать",
        "недополучить",
        "недоставать",
    }
)


@dataclass(frozen=True)
class NeVerbRule:
    spec: RuleSpec = RuleSpec(
        id="ne_verb",
        group="ne",
        scope="token",
        edit_type="split_join",
        mode="model_required",
        confidence=0.35,
        requires=NE_REQUIRES,
        description="Generate syntax-backed не + verb split candidates.",
    )

    def generate_candidates(self, token: str, context: RuleContext | None = None) -> Iterable[RuleCandidate]:
        word = token.lower()
        if not word.startswith("не") or len(word) <= 4:
            return []
        rest = word[2:]
        if word.startswith("недо") and is_known_word(word):
            return []
        if normal_forms(word) & NE_JOINED_NORMAL_FORMS:
            return []
        if not has_pos(rest, VERB_POSES):
            return []
        evidence = _token_evidence(context, "syntax/POS verb evidence")
        return [
            _candidate(
                self.spec,
                f"не {rest}",
                requires_model=True,
                syntax_family="ne_with_parts_of_speech",
                subtype="verb_split",
                trigger_text=token,
                trigger_pos="VERB",
                evidence=evidence,
            )
        ]

    def dirty_to_candidate(self, token: str, context: RuleContext | None = None) -> Iterable[RuleCandidate]:
        return self.generate_candidates(token, context)

    def generate(self, token: str, context: RuleContext | None = None) -> Iterable[RuleCandidate]:
        return self.generate_candidates(token, context)

    def generate_corruptions(
        self,
        text: str,
        tokens: tuple[Any, ...] | None = None,
        token_index: int | None = None,
        protected: tuple[tuple[int, int], ...] = (),
    ) -> Iterable[RuleCorruption]:
        word_tokens = _word_tokens(text, tokens)
        corruptions: list[RuleCorruption] = []
        for index, word in _iter_token_positions(word_tokens, token_index):
            if word.text.lower() != "не" or index + 1 >= len(word_tokens):
                continue
            next_word = word_tokens[index + 1]
            start, end = int(word.start), int(next_word.end)
            if _span_overlaps_protected(start, end, protected):
                continue
            if not has_pos(next_word.text, VERB_POSES):
                continue
            clean = text[start:end]
            dirty = "не" + next_word.text.lower()
            if not _same_rule_repairs(self, dirty, clean):
                continue
            corruptions.append(_corruption(self.spec, start, end, _match_case(clean, dirty), "ne_verb"))
        return corruptions

    def clean_to_dirty(
        self,
        text: str,
        tokens: tuple[Any, ...] | None = None,
        token_index: int | None = None,
        protected: tuple[tuple[int, int], ...] = (),
    ) -> Iterable[RuleCorruption]:
        return self.generate_corruptions(text, tokens, token_index, protected)


@dataclass(frozen=True)
class NePartOfSpeechRule:
    rule_id: str
    poses: frozenset[str]
    subtype_base: str
    description: str
    predicative_only: bool = False

    uses_rule_context: bool = True

    @property
    def spec(self) -> RuleSpec:
        return RuleSpec(
            id=self.rule_id,
            group="ne",
            scope="span",
            edit_type="split_join",
            mode="model_required",
            confidence=0.35,
            requires=NE_REQUIRES,
            description=self.description,
        )

    def generate_span(
        self,
        text: str,
        tokens: tuple[object, ...],
        token_index: int,
        context: RuleContext | None = None,
    ) -> Iterable[RuleEdit]:
        word_tokens = _word_tokens(text, tokens if tokens else None)
        if token_index < 0 or token_index >= len(word_tokens):
            return []

        token = word_tokens[token_index]
        word = str(getattr(token, "text", "")).lower()
        candidates: list[RuleEdit] = []

        if word.startswith("не") and len(word) > 4 and is_known_word(word):
            base = word[2:]
            if self._matches(base, word):
                candidates.append(self._edit(text, token, token, f"не {base}", f"{self.subtype_base}_split", context))

        if word == "не" and token_index + 1 < len(word_tokens):
            next_token = word_tokens[token_index + 1]
            base = str(getattr(next_token, "text", "")).lower()
            joined = f"не{base}"
            if self._matches(base, joined) and is_known_word(joined):
                candidates.append(self._edit(text, token, next_token, joined, f"{self.subtype_base}_join", context))

        return candidates

    def generate_corruptions(
        self,
        text: str,
        tokens: tuple[Any, ...] | None = None,
        token_index: int | None = None,
        protected: tuple[tuple[int, int], ...] = (),
    ) -> Iterable[RuleCorruption]:
        word_tokens = _word_tokens(text, tokens)
        corruptions: list[RuleCorruption] = []
        for index, token in _iter_token_positions(word_tokens, token_index):
            if _span_overlaps_protected(int(token.start), int(token.end), protected):
                continue
            word = str(getattr(token, "text", "")).lower()
            if word.startswith("не") and len(word) > 4 and is_known_word(word):
                base = word[2:]
                dirty = f"не {base}"
                if self._matches(base, word) and _span_rule_repairs(self, text, token, dirty):
                    corruptions.append(_token_corruption(self.spec, token, dirty, "ne_pos"))
                continue
            if word != "не" or index + 1 >= len(word_tokens):
                continue
            next_token = word_tokens[index + 1]
            start, end = int(token.start), int(next_token.end)
            if _span_overlaps_protected(start, end, protected):
                continue
            base = str(getattr(next_token, "text", "")).lower()
            dirty = f"не{base}"
            if self._matches(base, dirty) and is_known_word(dirty) and _span_rule_repairs_span(self, text, start, end, dirty):
                corruptions.append(_corruption(self.spec, start, end, _match_case(text[start:end], dirty), "ne_pos"))
        return corruptions

    def clean_to_dirty(
        self,
        text: str,
        tokens: tuple[Any, ...] | None = None,
        token_index: int | None = None,
        protected: tuple[tuple[int, int], ...] = (),
    ) -> Iterable[RuleCorruption]:
        return self.generate_corruptions(text, tokens, token_index, protected)

    def _matches(self, base: str, joined: str) -> bool:
        if self.predicative_only:
            return _has_predicative_parse(base) or _has_predicative_parse(joined)
        return _has_known_pos(base, self.poses)

    def _edit(self, text: str, first: Any, last: Any, replacement: str, subtype: str, context: RuleContext | None) -> RuleEdit:
        source = text[int(first.start) : int(last.end)]
        return _span_candidate(
            self.spec,
            text,
            first,
            last,
            replacement,
            syntax_family="ne_with_parts_of_speech",
            subtype=subtype,
            trigger_text=source,
            trigger_pos="/".join(sorted(self.poses)),
            evidence=_token_evidence(context, "syntax/POS part-of-speech evidence"),
        )


@dataclass(frozen=True)
class NiStableExpressionRule:
    spec: RuleSpec = RuleSpec(
        id="ni_stable_expression",
        group="ne_ni",
        scope="span",
        edit_type="split_join",
        mode="model_required",
        confidence=0.35,
        requires=NI_REQUIRES,
        description="Generate bounded не/ни candidates for stable constructions.",
    )

    uses_rule_context: bool = True

    def generate_span(
        self,
        text: str,
        tokens: tuple[object, ...],
        token_index: int,
        context: RuleContext | None = None,
    ) -> Iterable[RuleEdit]:
        word_tokens = _word_tokens(text, tokens if tokens else None)
        candidates: list[RuleEdit] = []
        for source, replacement in NI_STABLE_EXPRESSION_PAIRS:
            candidate = _phrase_span_candidate(
                self.spec,
                text,
                word_tokens,
                token_index,
                source,
                replacement,
                syntax_family="ni_context",
                subtype="stable_expression",
                evidence=_token_evidence(context, "bounded stable не/ни expression"),
            )
            if candidate:
                candidates.append(candidate)
        return candidates

    def generate_corruptions(
        self,
        text: str,
        tokens: tuple[Any, ...] | None = None,
        token_index: int | None = None,
        protected: tuple[tuple[int, int], ...] = (),
    ) -> Iterable[RuleCorruption]:
        del tokens, token_index
        return _phrase_corruptions_for_pairs(self, text, NI_STABLE_EXPRESSION_PAIRS, "ne_ni", protected)

    def clean_to_dirty(
        self,
        text: str,
        tokens: tuple[Any, ...] | None = None,
        token_index: int | None = None,
        protected: tuple[tuple[int, int], ...] = (),
    ) -> Iterable[RuleCorruption]:
        return self.generate_corruptions(text, tokens, token_index, protected)


@dataclass(frozen=True)
class NiParticleContextRule:
    spec: RuleSpec = RuleSpec(
        id="ni_particle_context",
        group="ne_ni",
        scope="token",
        edit_type="spelling",
        mode="model_required",
        confidence=0.35,
        requires=NI_REQUIRES,
        description="Generate bounded не/ни particle candidates in concessive contexts.",
    )

    def generate_candidates(self, token: str, context: RuleContext | None = None) -> Iterable[RuleCandidate]:
        if context is None or context.token_index < 0:
            return []
        word = token.lower()
        if word not in {"не", "ни"}:
            return []
        words = _word_tokens(context.text, context.tokens)
        if not _is_bounded_ni_particle_context(words, context.token_index):
            return []
        replacement = "ни" if word == "не" else "не"
        return [
            _candidate(
                self.spec,
                replacement,
                requires_model=True,
                syntax_family="ni_context",
                subtype="particle_context",
                trigger_text=token,
                trigger_pos="PART",
                evidence="bounded concessive не/ни particle context with syntax/POS evidence",
            )
        ]

    def dirty_to_candidate(self, token: str, context: RuleContext | None = None) -> Iterable[RuleCandidate]:
        return self.generate_candidates(token, context)

    def generate(self, token: str, context: RuleContext | None = None) -> Iterable[RuleCandidate]:
        return self.generate_candidates(token, context)


@dataclass(frozen=True)
class NnRule:
    rule_id: str
    description: str

    @property
    def spec(self) -> RuleSpec:
        return RuleSpec(
            id=self.rule_id,
            group="n_nn",
            scope="token",
            edit_type="spelling",
            mode="model_required",
            confidence=0.35,
            requires=N_NN_REQUIRES,
            description=self.description,
        )

    def generate_candidates(self, token: str, context: RuleContext | None = None) -> Iterable[RuleCandidate]:
        word = token.lower()
        candidates: list[RuleCandidate] = []
        for replacement in _n_nn_variants(word):
            if not is_known_word(replacement) or not self._matches(word, replacement):
                continue
            candidates.append(
                _candidate(
                    self.spec,
                    replacement,
                    requires_model=True,
                    syntax_family="n_nn_context",
                    subtype=self.rule_id,
                    trigger_text=token,
                    trigger_pos=_best_pos(replacement),
                    evidence=_token_evidence(context, "syntax/POS morphology evidence and dictionary-known replacement"),
                )
            )
        return candidates

    def dirty_to_candidate(self, token: str, context: RuleContext | None = None) -> Iterable[RuleCandidate]:
        return self.generate_candidates(token, context)

    def generate(self, token: str, context: RuleContext | None = None) -> Iterable[RuleCandidate]:
        return self.generate_candidates(token, context)

    def generate_corruptions(
        self,
        text: str,
        tokens: tuple[Any, ...] | None = None,
        token_index: int | None = None,
        protected: tuple[tuple[int, int], ...] = (),
    ) -> Iterable[RuleCorruption]:
        corruptions: list[RuleCorruption] = []
        for _index, token in _iter_token_positions(_word_tokens(text, tokens), token_index):
            if _span_overlaps_protected(int(token.start), int(token.end), protected):
                continue
            word = str(getattr(token, "text", "")).lower()
            for dirty in _n_nn_variants(word):
                if not _same_rule_repairs(self, dirty, word):
                    continue
                corruptions.append(_token_corruption(self.spec, token, dirty, self.spec.group))
        return corruptions

    def clean_to_dirty(
        self,
        text: str,
        tokens: tuple[Any, ...] | None = None,
        token_index: int | None = None,
        protected: tuple[tuple[int, int], ...] = (),
    ) -> Iterable[RuleCorruption]:
        return self.generate_corruptions(text, tokens, token_index, protected)

    def _matches(self, source: str, replacement: str) -> bool:
        if self.rule_id == "n_nn_short_form":
            return _is_short_n_nn_form(replacement)
        if self.rule_id == "n_nn_participle":
            return _has_known_pos(replacement, N_NN_PARTICIPLE_POSES) and not _is_short_n_nn_form(replacement)
        if self.rule_id == "n_nn_deverbal_adjective":
            return _has_known_pos(source, N_NN_PARTICIPLE_POSES) and _has_known_pos(replacement, N_NN_ADJECTIVE_POSES)
        if self.rule_id == "n_nn_adjective":
            return _has_known_pos(replacement, N_NN_ADJECTIVE_POSES) and not _has_known_pos(
                replacement,
                N_NN_PARTICIPLE_POSES,
            )
        return False


@dataclass(frozen=True)
class TsyaSoftDeleteRule:
    spec: RuleSpec = RuleSpec(
        id="tsya_soft_delete",
        group="tsya",
        scope="token",
        edit_type="spelling",
        mode="model_required",
        confidence=0.9,
        requires=NE_REQUIRES,
        description="Generate -ться -> -тся candidate for verb forms.",
    )

    def generate_candidates(self, token: str, context: RuleContext | None = None) -> Iterable[RuleCandidate]:
        word = token.lower()
        if not word.endswith("ться"):
            return []
        replacement = word[: -len("ться")] + "тся"
        if not (is_known_word(replacement) and has_pos(replacement, TSYA_CONTEXT_POSES)):
            return []
        return [
            _candidate(
                self.spec,
                replacement,
                requires_model=True,
                syntax_family="tsya_tsya_context",
                subtype="soft_delete",
                trigger_text=token,
                trigger_pos="VERB",
                evidence=_token_evidence(context, "syntax/POS verb context evidence"),
            )
        ]

    def dirty_to_candidate(self, token: str, context: RuleContext | None = None) -> Iterable[RuleCandidate]:
        return self.generate_candidates(token, context)

    def generate(self, token: str, context: RuleContext | None = None) -> Iterable[RuleCandidate]:
        return self.generate_candidates(token, context)

    def generate_corruptions(
        self,
        text: str,
        tokens: tuple[Any, ...] | None = None,
        token_index: int | None = None,
        protected: tuple[tuple[int, int], ...] = (),
    ) -> Iterable[RuleCorruption]:
        corruptions: list[RuleCorruption] = []
        for _index, token in _iter_token_positions(_word_tokens(text, tokens), token_index):
            if _span_overlaps_protected(int(token.start), int(token.end), protected):
                continue
            word = token.text.lower()
            if not word.endswith("тся"):
                continue
            dirty = word[: -len("тся")] + "ться"
            if _same_rule_repairs(self, dirty, word):
                corruptions.append(_token_corruption(self.spec, token, dirty, self.spec.group))
        return corruptions

    def clean_to_dirty(
        self,
        text: str,
        tokens: tuple[Any, ...] | None = None,
        token_index: int | None = None,
        protected: tuple[tuple[int, int], ...] = (),
    ) -> Iterable[RuleCorruption]:
        return self.generate_corruptions(text, tokens, token_index, protected)


@dataclass(frozen=True)
class TsyaSoftInsertRule:
    spec: RuleSpec = RuleSpec(
        id="tsya_soft_insert",
        group="tsya",
        scope="token",
        edit_type="spelling",
        mode="model_required",
        confidence=0.9,
        requires=NE_REQUIRES,
        description="Generate -тся -> -ться candidate for verb forms.",
    )

    def generate_candidates(self, token: str, context: RuleContext | None = None) -> Iterable[RuleCandidate]:
        word = token.lower()
        if not word.endswith("тся"):
            return []
        replacement = word[: -len("тся")] + "ться"
        if not (is_known_word(replacement) and has_pos(replacement, TSYA_CONTEXT_POSES)):
            return []
        return [
            _candidate(
                self.spec,
                replacement,
                requires_model=True,
                syntax_family="tsya_tsya_context",
                subtype="soft_insert",
                trigger_text=token,
                trigger_pos="VERB",
                evidence=_token_evidence(context, "syntax/POS verb context evidence"),
            )
        ]

    def dirty_to_candidate(self, token: str, context: RuleContext | None = None) -> Iterable[RuleCandidate]:
        return self.generate_candidates(token, context)

    def generate(self, token: str, context: RuleContext | None = None) -> Iterable[RuleCandidate]:
        return self.generate_candidates(token, context)

    def generate_corruptions(
        self,
        text: str,
        tokens: tuple[Any, ...] | None = None,
        token_index: int | None = None,
        protected: tuple[tuple[int, int], ...] = (),
    ) -> Iterable[RuleCorruption]:
        corruptions: list[RuleCorruption] = []
        for _index, token in _iter_token_positions(_word_tokens(text, tokens), token_index):
            if _span_overlaps_protected(int(token.start), int(token.end), protected):
                continue
            word = token.text.lower()
            if not word.endswith("ться"):
                continue
            dirty = word[: -len("ться")] + "тся"
            if _same_rule_repairs(self, dirty, word):
                corruptions.append(_token_corruption(self.spec, token, dirty, self.spec.group))
        return corruptions

    def clean_to_dirty(
        self,
        text: str,
        tokens: tuple[Any, ...] | None = None,
        token_index: int | None = None,
        protected: tuple[tuple[int, int], ...] = (),
    ) -> Iterable[RuleCorruption]:
        return self.generate_corruptions(text, tokens, token_index, protected)


@dataclass(frozen=True)
class ContextPairRule:
    rule_id: str
    pairs: tuple[tuple[str, str], ...]

    uses_rule_context: bool = True

    @property
    def spec(self) -> RuleSpec:
        return RuleSpec(
            id=self.rule_id,
            group="context_split_join",
            scope="span",
            edit_type="split_join",
            mode="model_required",
            confidence=0.35,
            requires=CONTEXT_REQUIRES,
            description="Context-dependent split/join candidates that need model confirmation.",
        )

    def generate_span(
        self,
        text: str,
        tokens: tuple[object, ...],
        token_index: int,
        context: RuleContext | None = None,
    ) -> Iterable[RuleEdit]:
        candidates: list[RuleEdit] = []
        word_tokens = _word_tokens(text, tokens if tokens else None)
        for source, replacement in self.pairs:
            candidate = _phrase_span_candidate(
                self.spec,
                text,
                word_tokens,
                token_index,
                source,
                replacement,
                syntax_family="context_pairs",
                subtype=self.rule_id.removeprefix("context_"),
                evidence=_token_evidence(context, "bounded context pair with local syntax evidence"),
            )
            if candidate:
                candidates.append(candidate)
        return candidates

    def generate_corruptions(
        self,
        text: str,
        tokens: tuple[Any, ...] | None = None,
        token_index: int | None = None,
        protected: tuple[tuple[int, int], ...] = (),
    ) -> Iterable[RuleCorruption]:
        del tokens, token_index
        return _phrase_corruptions_for_pairs(self, text, self.pairs, "context_pairs", protected)

    def clean_to_dirty(
        self,
        text: str,
        tokens: tuple[Any, ...] | None = None,
        token_index: int | None = None,
        protected: tuple[tuple[int, int], ...] = (),
    ) -> Iterable[RuleCorruption]:
        return self.generate_corruptions(text, tokens, token_index, protected)


def syntax_orthography_rules() -> tuple[object, ...]:
    return (
        NnRule("n_nn_short_form", "Generate model-scored Н/НН candidates for short adjective and participle forms."),
        NnRule("n_nn_participle", "Generate model-scored Н/НН candidates for participles."),
        NnRule("n_nn_deverbal_adjective", "Generate model-scored Н/НН candidates for deverbal adjectives."),
        NnRule("n_nn_adjective", "Generate model-scored Н/НН candidates for adjectives."),
        NeVerbRule(),
        NePartOfSpeechRule("ne_predicative", NE_PREDICATIVE_POSES, "predicative", "Generate не + predicative split/join candidates.", True),
        NePartOfSpeechRule("ne_participle", NE_PARTICIPLE_POSES, "participle", "Generate не + participle split/join candidates."),
        NePartOfSpeechRule("ne_adverb", NE_ADVERB_POSES, "adverb", "Generate не + adverb split/join candidates."),
        NePartOfSpeechRule("ne_short_form", NE_SHORT_FORM_POSES, "short_form", "Generate не + short-form adjective split/join candidates."),
        NePartOfSpeechRule("ne_adjective", NE_FULL_ADJECTIVE_POSES, "adjective", "Generate не + adjective split/join candidates."),
        NiStableExpressionRule(),
        NiParticleContextRule(),
        TsyaSoftDeleteRule(),
        TsyaSoftInsertRule(),
        ContextPairRule("context_tak_zhe", (("также", "так же"), ("так же", "также"))),
        ContextPairRule("context_to_zhe", (("тоже", "то же"), ("то же", "тоже"))),
        ContextPairRule("context_chto_by", (("чтобы", "что бы"), ("что бы", "чтобы"))),
        ContextPairRule("context_za_to", (("зато", "за то"), ("за то", "зато"))),
        ContextPairRule("context_vsledstvie", (("вследствие", "в следствие"), ("в следствие", "вследствие"))),
        ContextPairRule(
            "context_nesmotrya",
            (
                ("несмотря на", "не смотря на"),
                ("не смотря на", "несмотря на"),
                ("несмотря", "не смотря"),
                ("не смотря", "несмотря"),
            ),
        ),
    )


def _candidate(
    spec: RuleSpec,
    replacement: str,
    *,
    requires_model: bool,
    syntax_family: str,
    subtype: str,
    trigger_text: str,
    trigger_pos: str,
    evidence: str,
    edit_type: str | None = None,
    mode: RuleMode | None = None,
) -> RuleCandidate:
    candidate_mode = mode or spec.mode
    metadata = _metadata(
        syntax_family=syntax_family,
        subtype=subtype,
        trigger_text=trigger_text,
        trigger_pos=trigger_pos,
        evidence=evidence,
    )
    return RuleCandidate(
        replacement=replacement,
        edit_type=edit_type or spec.edit_type,
        confidence=spec.confidence,
        requires_model=requires_model or candidate_mode in {"candidate_only", "model_required"},
        rule_id=spec.id,
        mode=candidate_mode,
        group=spec.group,
        requires=spec.requires,
        edit_domain=ORTHOGRAPHY_DOMAIN,
        syntax_family=syntax_family,
        subtype=subtype,
        trigger_text=trigger_text,
        trigger_lemma=trigger_text.lower(),
        trigger_pos=trigger_pos,
        evidence=evidence,
        confidence_source="syntax_orthography_rule",
        implementation_group=_implementation_group(syntax_family),
        metadata=metadata,
    )


def _span_candidate(
    spec: RuleSpec,
    text: str,
    first: Any,
    last: Any,
    replacement: str,
    *,
    syntax_family: str,
    subtype: str,
    trigger_text: str,
    trigger_pos: str = "",
    evidence: str,
) -> RuleEdit:
    start = int(first.start)
    end = int(last.end)
    source = text[start:end]
    metadata = _metadata(
        syntax_family=syntax_family,
        subtype=subtype,
        trigger_text=trigger_text,
        trigger_pos=trigger_pos,
        evidence=evidence,
    )
    return RuleEdit(
        source=source,
        replacement=_match_case(source, replacement),
        edit_type=spec.edit_type,
        start=start,
        end=end,
        confidence=spec.confidence,
        requires_model=True,
        rule_id=spec.id,
        mode=spec.mode,
        group=spec.group,
        requires=spec.requires,
        edit_domain=ORTHOGRAPHY_DOMAIN,
        syntax_family=syntax_family,
        subtype=subtype,
        trigger_text=trigger_text,
        trigger_lemma=trigger_text.lower(),
        trigger_pos=trigger_pos,
        evidence=evidence,
        confidence_source="syntax_orthography_rule",
        implementation_group=_implementation_group(syntax_family),
        metadata=metadata,
    )


def _phrase_span_candidate(
    spec: RuleSpec,
    text: str,
    tokens: tuple[Any, ...],
    token_index: int,
    source: str,
    replacement: str,
    *,
    syntax_family: str,
    subtype: str,
    evidence: str,
) -> RuleEdit | None:
    source_words = source.split()
    end_index = token_index + len(source_words)
    if token_index < 0 or end_index > len(tokens):
        return None
    span_words = tokens[token_index:end_index]
    if [str(getattr(word, "text", "")).lower() for word in span_words] != source_words:
        return None
    start = int(getattr(span_words[0], "start"))
    end = int(getattr(span_words[-1], "end"))
    if text[start:end].lower().split() != source_words:
        return None
    return _span_candidate(
        spec,
        text,
        span_words[0],
        span_words[-1],
        replacement,
        syntax_family=syntax_family,
        subtype=subtype,
        trigger_text=text[start:end],
        evidence=evidence,
    )


def _metadata(
    *,
    syntax_family: str,
    subtype: str,
    trigger_text: str,
    trigger_pos: str,
    evidence: str,
) -> dict[str, str]:
    return {
        "edit_domain": ORTHOGRAPHY_DOMAIN,
        "syntax_family": syntax_family,
        "subtype": subtype,
        "trigger_text": trigger_text,
        "trigger_lemma": trigger_text.lower(),
        "trigger_pos": trigger_pos,
        "evidence": evidence,
        "confidence_source": "syntax_orthography_rule",
        "implementation_group": _implementation_group(syntax_family),
    }


def _implementation_group(syntax_family: str) -> str:
    return {
        "ne_with_parts_of_speech": "syntax_ne_split_join",
        "ni_context": "syntax_ni_context",
        "n_nn_context": "syntax_n_nn_context",
        "tsya_tsya_context": "syntax_tsya_context",
        "context_pairs": "syntax_context_pairs",
    }.get(syntax_family, f"syntax_{syntax_family}")


def _token_evidence(context: RuleContext | None, fallback: str) -> str:
    if context is not None and context.syntax_tokens:
        return f"{fallback}; syntax tokens available"
    return fallback


def _is_bounded_ni_particle_context(tokens: tuple[Any, ...], token_index: int) -> bool:
    lowered = [str(getattr(token, "text", "")).lower() for token in tokens]
    if token_index < 0 or token_index >= len(lowered):
        return False
    left = lowered[max(0, token_index - 4) : token_index]
    if not ("что" in left and "бы" in left):
        return False
    right = lowered[token_index + 1 : min(len(lowered), token_index + 4)]
    return any(has_pos(word, VERB_POSES) for word in right)


def _span_rule_repairs(rule: object, clean_text: str, clean_token: Any, dirty_replacement: str) -> bool:
    start = int(clean_token.start)
    end = int(clean_token.end)
    return _span_rule_repairs_span(rule, clean_text, start, end, dirty_replacement)


def _span_rule_repairs_span(rule: object, clean_text: str, start: int, end: int, dirty_replacement: str) -> bool:
    clean = clean_text[start:end]
    dirty = _match_case(clean, dirty_replacement)
    dirty_text = clean_text[:start] + dirty + clean_text[end:]
    for candidate in _span_candidates_for_rule(rule, dirty_text, None, None):
        if candidate.rule_id == rule.spec.id and candidate.replacement.lower() == clean.lower():
            return True
    return False


def _span_candidates_for_rule(
    rule: object,
    text: str,
    tokens: tuple[Any, ...] | None,
    token_index: int | None,
) -> Iterable[RuleEdit]:
    word_tokens = _word_tokens(text, tokens if tokens else None)
    indexes = range(len(word_tokens)) if token_index is None else (token_index,)
    candidates: list[RuleEdit] = []
    for index in indexes:
        if 0 <= index < len(word_tokens):
            candidates.extend(rule.generate_span(text, word_tokens, index))
    return candidates


def _phrase_corruptions_for_pairs(
    rule: object,
    text: str,
    pairs: tuple[tuple[str, str], ...],
    group: str,
    protected: tuple[tuple[int, int], ...],
) -> list[RuleCorruption]:
    corruptions: list[RuleCorruption] = []
    lower = text.lower()
    seen: set[tuple[int, int, str]] = set()
    for dirty, clean in pairs:
        start = lower.find(clean)
        if start < 0:
            continue
        end = start + len(clean)
        if _span_overlaps_protected(start, end, protected):
            continue
        key = (start, end, dirty)
        if key in seen or not _span_rule_repairs_span(rule, text, start, end, dirty):
            continue
        seen.add(key)
        corruptions.append(_corruption(rule.spec, start, end, _match_case(text[start:end], dirty), group))
    return corruptions


def _word_tokens(text: str, tokens: tuple[Any, ...] | None) -> tuple[Any, ...]:
    return tokens if tokens is not None else tuple(tokenize_words(text))


def _iter_token_positions(tokens: tuple[Any, ...], token_index: int | None) -> Iterable[tuple[int, Any]]:
    if token_index is None:
        return tuple(enumerate(tokens))
    if 0 <= token_index < len(tokens):
        return ((token_index, tokens[token_index]),)
    return ()


def _same_rule_repairs(rule: object, dirty: str, clean: str) -> bool:
    generator = getattr(rule, "generate_candidates", getattr(rule, "generate", None))
    if not generator:
        return False
    for candidate in generator(dirty, RuleContext()):
        if candidate.rule_id == rule.spec.id and candidate.replacement.lower() == clean.lower():
            return True
    return False


def _corruption(spec: RuleSpec, start: int, end: int, replacement: str, group: str) -> RuleCorruption:
    return RuleCorruption(
        start=start,
        end=end,
        replacement=replacement,
        error_type=spec.edit_type,
        group=group,
        rule_id=spec.id,
    )


def _token_corruption(spec: RuleSpec, token: Any, dirty: str, group: str) -> RuleCorruption:
    return _corruption(spec, int(token.start), int(token.end), _match_case(str(token.text), dirty), group)


def _span_overlaps_protected(start: int, end: int, protected: tuple[tuple[int, int], ...]) -> bool:
    return any(start < protected_end and protected_start < end for protected_start, protected_end in protected)


def _n_nn_variants(word: str) -> tuple[str, ...]:
    variants: list[str] = []
    seen: set[str] = set()
    for index, char in enumerate(word):
        if char != "н":
            continue
        if word[index : index + 2] == "нн":
            variant = word[:index] + "н" + word[index + 2 :]
        elif (index == 0 or word[index - 1] != "н") and (index + 1 == len(word) or word[index + 1] != "н"):
            variant = word[:index] + "нн" + word[index + 1 :]
        else:
            continue
        if variant != word and variant not in seen:
            seen.add(variant)
            variants.append(variant)
    return tuple(variants)


def _has_known_pos(word: str, poses: frozenset[str]) -> bool:
    return any(getattr(parse, "is_known", False) and getattr(parse.tag, "POS", None) in poses for parse in parses(word))


def _has_predicative_parse(word: str) -> bool:
    for parse in parses(word):
        if not getattr(parse, "is_known", False):
            continue
        pos = getattr(parse.tag, "POS", None)
        tag_text = str(parse.tag)
        if pos == "PRED" or "Prdx" in tag_text:
            return True
    return False


def _is_short_n_nn_form(word: str) -> bool:
    return _has_known_pos(word, frozenset({"ADJS", "PRTS"}))


def _best_pos(word: str) -> str:
    known = [parse for parse in parses(word) if getattr(parse, "is_known", False)]
    if not known:
        return ""
    best = max(known, key=lambda parse: float(getattr(parse, "score", 0.0)))
    return str(getattr(best.tag, "POS", "") or "")


def _match_case(source: str, replacement: str) -> str:
    if source[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement
