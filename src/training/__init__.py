__all__ = ["train"]


def train(*args, **kwargs):
    from src.training.train import train as _train

    return _train(*args, **kwargs)
