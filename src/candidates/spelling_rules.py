from src.candidates.frequent_errors import WRONG_TO_CORRECT


def spelling_candidates(word: str) -> list[str]:
    candidate = WRONG_TO_CORRECT.get(word.lower())
    return [candidate] if candidate else []
