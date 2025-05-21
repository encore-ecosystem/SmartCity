from cvtk.bbox import Bbox_2xy
from ..base.yolo import YoloBase
from pathlib import Path
from PIL.Image import Image


class YoloCarDetector(YoloBase):
    def predict(self, inp: Image) -> list[Bbox_2xy]:
        return self._predict_bboxes(inp)
    
    @classmethod
    def load(cls, path: Path) -> "YoloCarDetector":
        weights_path = path / cls.weights_name()
        return YoloCarDetector(weights_path)
    
    def save(self, path: Path):
        weights_path = path / self.weights_name()
        self._model.save(weights_path)

    @staticmethod
    def weights_name() -> str:
        return "yolo_car_detector.pt"
    

__all__ = [
    "YoloCarDetector",
]
