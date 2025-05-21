from experimentator import Model, Measurer
from cvtk.interfaces.dataset import AbstractDataset
from prologger import Logger
from pathlib import Path
from ultralytics import YOLO
from ultralytics.models.fastsam import predict
from PIL.Image import Image
from cvtk.bbox import Bbox_2xy


class YoloBase(Model):
    def __init__(self, weights_path: Path):
        assert weights_path.exists()
        self._model = YOLO(model=weights_path)
    
    def _predict_bboxes(self, image: Image):
        width = image.width
        height = image.height

        results = self._model.predict(source=image, verbose=False)
        bboxes= []
        for result in results:
            for bbox_data in result.boxes.data.tolist():
                bbox = Bbox_2xy(
                    points     = [bbox_data[0] / width, bbox_data[1] / height, bbox_data[2] / width, bbox_data[3] / height],
                    value      = result.names[int(bbox_data[5])],
                    category   = int(bbox_data[5]),
                    confidence = bbox_data[4],
                )
                bboxes.append(bbox)
        return bboxes
    
    def is_loses_information(self) -> bool:
        return True

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
