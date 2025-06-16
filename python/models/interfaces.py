from typing import Protocol, runtime_checkable, Mapping, Any, Iterator, Dict
import torch


@runtime_checkable
class TrainableModel(Protocol):
    def forward(self, x: Any) -> torch.Tensor:
        ...

    def __call__(self, x: Any) -> torch.Tensor:
        ...

    def parameters(self) -> Iterator[torch.nn.Parameter]:
        ...

    def train(self, mode: bool = True) -> torch.nn.Module:
        ...

    def eval(self) -> torch.nn.Module:
        ...

    def state_dict(self, *args, **kwargs) -> Dict[str, Any]:
        ...

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False
    ) -> Any:
        ...

    def to(self, device: torch.device) -> torch.nn.Module:
        ...
