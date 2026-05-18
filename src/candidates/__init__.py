__all__ = ["Candidate", "CandidateGenerator"]


def __getattr__(name: str):
    if name in __all__:
        from src.candidates.candidate_generator import Candidate, CandidateGenerator

        return {"Candidate": Candidate, "CandidateGenerator": CandidateGenerator}[name]
    raise AttributeError(name)
