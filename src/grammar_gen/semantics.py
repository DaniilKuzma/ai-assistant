from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.grammar_gen.randomness import RandomSource


@dataclass(frozen=True)
class SemanticClass:
    name: str


@dataclass(frozen=True)
class PrepSlot:
    preposition: str
    case: str
    semantic_class: str


@dataclass(frozen=True)
class VerbFrame:
    frame_id: str
    verb_lemma: str
    subject_classes: tuple[str, ...]
    object_classes: tuple[str, ...]
    prep_slots: tuple[PrepSlot, ...]
    allow_ne: bool
    frame_family: str


@dataclass(frozen=True)
class SemanticFrameLexicon:
    frames: tuple[VerbFrame, ...]

    @classmethod
    def from_dir(cls, path: str | Path) -> "SemanticFrameLexicon":
        frame_path = Path(path) / "verb_frames.csv"
        if not frame_path.exists():
            return _fallback_frame_lexicon()

        try:
            frames = tuple(_frame_from_row(row) for row in _read_csv(frame_path))
        except (KeyError, ValueError, OSError, csv.Error):
            return _fallback_frame_lexicon()

        if not frames:
            return _fallback_frame_lexicon()
        return cls(frames=frames)

    def random_frame(
        self,
        rng: RandomSource,
        frame_family: str | None = None,
        allow_object: bool | None = None,
        allow_ne: bool | None = None,
    ) -> VerbFrame:
        candidates = [
            frame
            for frame in self.frames
            if (frame_family is None or frame.frame_family == frame_family)
            and (allow_object is None or bool(frame.object_classes) is allow_object)
            and (allow_ne is None or frame.allow_ne is allow_ne)
        ]
        if not candidates:
            raise ValueError("No semantic frames match the requested filters.")
        return rng.choice(tuple(candidates))

    def frames_for_verb(self, lemma: str) -> tuple[VerbFrame, ...]:
        return tuple(frame for frame in self.frames if frame.verb_lemma == lemma)

    def validate_subject(self, frame: VerbFrame, noun_entry: Any) -> bool:
        semantic_class = _noun_class(noun_entry)
        if semantic_class not in frame.subject_classes:
            return False
        if _requires_agentive_subject(frame) and not bool(getattr(noun_entry, "agentive", False)):
            return False
        return True

    def validate_object(self, frame: VerbFrame, noun_entry: Any) -> bool:
        semantic_class = _noun_class(noun_entry)
        if semantic_class not in frame.object_classes:
            return False
        return bool(getattr(noun_entry, "can_be_patient", False))

    def validate_prep_slot(self, frame: VerbFrame, prep: str, noun_entry: Any, case: str) -> bool:
        normalized_case = _canonical_case(case)
        for slot in frame.prep_slots:
            if (
                slot.preposition == prep
                and _canonical_case(slot.case) == normalized_case
                and slot.semantic_class == _noun_class(noun_entry)
            ):
                return _noun_fits_prep_role(noun_entry)
        return False


def _frame_from_row(row: dict[str, str]) -> VerbFrame:
    return VerbFrame(
        frame_id=row["frame_id"],
        verb_lemma=row["verb_lemma"],
        subject_classes=_parse_classes(row["subject_classes"]),
        object_classes=_parse_classes(row["object_classes"]),
        prep_slots=_parse_prep_slots(row["prep_slots"]),
        allow_ne=_parse_bool(row["allow_ne"]),
        frame_family=row["frame_family"],
    )


def _parse_classes(value: str) -> tuple[str, ...]:
    if not value.strip():
        return ()
    return tuple(part.strip() for part in value.split("|") if part.strip())


def _parse_prep_slots(value: str) -> tuple[PrepSlot, ...]:
    if not value.strip():
        return ()
    slots: list[PrepSlot] = []
    for raw_slot in value.split("|"):
        parts = [part.strip() for part in raw_slot.split(":")]
        if len(parts) != 3 or not all(parts):
            raise ValueError(f"Invalid prep slot {raw_slot!r}.")
        slots.append(PrepSlot(preposition=parts[0], case=parts[1], semantic_class=parts[2]))
    return tuple(slots)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [
            {key: (value or "").strip() for key, value in row.items() if key is not None}
            for row in csv.DictReader(handle)
            if any((value or "").strip() for value in row.values())
        ]


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"Expected true or false, got {value!r}.")


def _canonical_case(case: str) -> str:
    return {
        "nom": "nomn",
        "nomn": "nomn",
        "gen": "gent",
        "gent": "gent",
        "dat": "datv",
        "datv": "datv",
        "acc": "accs",
        "accs": "accs",
        "ins": "ablt",
        "ablt": "ablt",
        "prep": "loct",
        "loct": "loct",
    }.get(case, case)


def _noun_class(noun_entry: Any) -> str:
    return str(getattr(noun_entry, "semantic_class"))


def _requires_agentive_subject(frame: VerbFrame) -> bool:
    return frame.frame_family in {
        "action",
        "approval",
        "biological_action",
        "buying",
        "calculation",
        "communication",
        "comparison",
        "correction",
        "data",
        "document_work",
        "event_work",
        "file_action",
        "meeting_work",
        "movement",
        "perception",
        "placement",
        "problem_solving",
        "publication",
        "reading",
        "storage",
    }


def _noun_fits_prep_role(noun_entry: Any) -> bool:
    semantic_class = _noun_class(noun_entry)
    if semantic_class in {"place", "surface", "building"}:
        return bool(getattr(noun_entry, "can_be_location", False))
    return True


def _fallback_frame_lexicon() -> SemanticFrameLexicon:
    return SemanticFrameLexicon(
        frames=(
            VerbFrame(
                "buy_goods",
                "купить",
                ("person", "organization"),
                ("food", "goods", "object", "property", "service"),
                (),
                True,
                "buying",
            ),
            VerbFrame(
                "contain_info",
                "содержать",
                ("document", "text", "report", "law"),
                ("information", "error", "requirement", "fact"),
                (),
                False,
                "content",
            ),
            VerbFrame(
                "stand_place",
                "стоять",
                ("building", "object", "vehicle"),
                (),
                (
                    PrepSlot("в", "loct", "place"),
                    PrepSlot("на", "loct", "surface"),
                    PrepSlot("у", "gent", "place"),
                ),
                True,
                "location",
            ),
        )
    )
