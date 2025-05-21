from models.recognizer import YoloOCR
from pathlib import Path


def main():
    ocr_path = Path().resolve() / "weights" / "recognizer" / "best.pt"
    ocr = YoloOCR(ocr_path)
    text = ocr.predict(Path().resolve() / "datasets/ocrv1/train/images/3_png.rf.3ea40e19285147fafb9fb184946081aa.jpg")
    print(text)


if __name__ == "__main__":
    main()
