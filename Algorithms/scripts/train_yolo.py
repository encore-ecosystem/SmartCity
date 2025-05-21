from ultralytics import YOLO

YAML_PATH = "./datasets/vehicle_sparced/data.yaml"


def main():
    model = YOLO("./weights/detector/yolov5nu.pt")
    model.train(data=YAML_PATH, epochs=100, imgsz=640, device=0)


if __name__ == "__main__":
    main()
