from cvtk.bbox import Bbox_2xy
from ..base.yolo import YoloBase
from pathlib import Path
from PIL.Image import Image


class YoloPlateDetector(YoloBase):
    @classmethod
    def load(cls, path: Path) -> "YoloPlateDetector":
        weights_path = path / cls.weights_name()
        return cls(weights_path)
    
    def save(self, path: Path):
        weights_path = path / self.weights_name()
        self._model.save(weights_path)

    @staticmethod
    def weights_name() -> str:
        return "yolo_plate_detector.pt"

    def predict(self, inp: Image) -> list[Bbox_2xy]:
        return self._predict_bboxes(inp)


__all__ = [
    "YoloPlateDetector",
]
