from typing import Protocol
import torch


class TrainableModel(Protocol):
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        ...

    def parameters(self):
        ...

    def train(self, mode: bool = True):
        ...

    def eval(self):
        ...

    def state_dict(self):
        ...

    def load_state_dict(self, state_dict):
        ...
