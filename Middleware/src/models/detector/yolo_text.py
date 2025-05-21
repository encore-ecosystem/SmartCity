from ..base.yolo import YoloBase
from pathlib import Path
from PIL.Image import Image


class YoloTextDetector(YoloBase):
    @classmethod
    def load(cls, path: Path) -> "YoloTextDetector":
        weights_path = path / cls.weights_name()
        return cls(weights_path)
    
    def save(self, path: Path):
        weights_path = path / self.weights_name()
        self._model.save(weights_path)

    @staticmethod
    def weights_name() -> str:
        return "yolo_text_detector.pt"

    def predict(self, inp: Image) -> list[str]:
        bboxes = self._predict_bboxes(inp)

        bboxes.sort(key=lambda x: x.bbox[1], reverse=True)
        lines = []
        current_line = []
        while bboxes:
            bbox = bboxes.pop()

            if current_line:
                if abs(current_line[-1].bbox[1] - bbox.bbox[1]) <= 0.4:
                    current_line.append(bbox)
                else:
                    lines.append(current_line)
                    current_line = [bbox]
            else:
                current_line.append(bbox)
        else:
            lines.append(current_line)

        result = []
        for line in lines:
            line.sort(key=lambda x: x.bbox[0])
            text = "".join(x.value for x in line)
            result.append(text)

        return result


__all__ = [
    "YoloTextDetector",
]
