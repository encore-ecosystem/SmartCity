from cvtk.bbox import Bbox_2xy
from experimentator import Pipeline
from pathlib import Path
from src.models.detector import YoloPlateDetector, YoloTextDetector
from PIL import Image


class PlateTextPipeline(Pipeline):
    @classmethod
    def load(cls, src_path: Path) -> 'PlateTextPipeline':
        models = [
            YoloPlateDetector.load(src_path / "plate_det"),
            YoloTextDetector.load(src_path / "text_det")
        ]
        return PlateTextPipeline(models)

    def save(self, dst_path: Path):
        self._models[0].save(dst_path / "plate_det")
        self._models[1].save(dst_path / "text_det")
    
    def predict(self, image_path: Path) -> list[Bbox_2xy]:
        image = Image.open(image_path)
        plate_bboxes = self._models[0].predict(image)
        for bbox in plate_bboxes:
            crop = bbox.crop_on(image)
            bbox.value = "".join(self._models[1].predict(crop))
        return plate_bboxes


__all__ = [
    "PlateTextPipeline",
]
