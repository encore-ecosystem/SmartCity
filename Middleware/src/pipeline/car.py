from experimentator import Pipeline
from pathlib import Path
from src.models.detector import YoloCarDetector
from PIL import Image


class CarPipeline(Pipeline):
    @classmethod
    def load(cls, src_path: Path) -> 'CarPipeline':
        models = [
            YoloCarDetector.load(src_path / "car_det")
        ]
        return CarPipeline(models)

    def save(self, dst_path: Path):
        path = dst_path / "car_det"
        path.mkdir(exist_ok=True)
        self._models[0].save(path)
    
    def predict(self, image_path: Path):
        image = Image.open(image_path)
        res = self._models[0].predict(image)
        return res


__all__ = [
    "CarPipeline",
]
