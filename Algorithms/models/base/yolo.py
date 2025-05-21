from experimentator import Model, Measurer
from cvtk.interfaces.dataset import AbstractDataset
from prologger import Logger
from pathlib import Path
from ultralytics import YOLO
from ultralytics.models.fastsam import predict


class YoloBase(Model):
    def __init__(self, weights_path: Path):
        assert weights_path.exists()
        self._model = YOLO(model=weights_path)
    
    def is_loses_information(self) -> bool:
        return True
    
    @classmethod
    def load(cls, weights_path: Path):
        raise NotImplementedError()

    def save(self, weights_path: Path):
        raise NotImplementedError()

    def train_step(self, dataset: AbstractDataset, measurer: Measurer, logger: Logger, current_epoch: int):
        raise NotImplementedError

    def test_step(self, dataset: AbstractDataset, measurer: Measurer, logger: Logger, current_epoch: int):
        raise NotImplementedError

    def eval_step(self, dataset: AbstractDataset, measurer: Measurer, logger: Logger, current_epoch: int):
        raise NotImplementedError
    
    compute = predict



__all__ = [
    "YoloBase"
]
