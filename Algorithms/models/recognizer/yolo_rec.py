from ..base.yolo import YoloBase
from pathlib import Path
from PIL import Image
from cvtk.bbox import Bbox_2xy


class YoloOCR(YoloBase):
    def predict(self, image_path: Path) -> list[str]:
        image = Image.open(image_path)
        width = image.width
        height = image.height

        results = self._model.predict(source=image, verbose=False)
        bboxes= []
        for result in results:
            for bbox_data in result.boxes.data.tolist():
                bbox      = Bbox_2xy(
                    points     = [bbox_data[0] / width, bbox_data[1] / height, bbox_data[2] / width, bbox_data[3] / height],
                    value      = result.names[int(bbox_data[5])],
                    category   = int(bbox_data[5]),
                    confidence = bbox_data[4],
                )
                bboxes.append(bbox)

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
    "YoloOCR",
]
