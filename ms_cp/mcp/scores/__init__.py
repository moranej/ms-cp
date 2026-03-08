from .aps import APSScore
from .lac import LACScore
from .raps import RAPSScore


def get_score_method(name: str):
    name = name.lower()
    if name == "lac":
        return LACScore()
    if name == "aps":
        return APSScore()
    if name == "raps":
        return RAPSScore()
    raise ValueError(f"Unknown score: {name}")
