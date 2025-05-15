from src import EXPERIMENTS_PATH
from experimentator import ExperimentatorClient
from pathlib import Path
from tqdm import tqdm
import os


def main():
    input_path = Path(input("Enter path to rendered images: "))

    # guards
    if not input_path.exists():
        raise ValueError("Path does not exists")
    if not input_path.is_dir():
        raise ValueError("Path should be directory")

    # make / load experimentator pipeline
    client     = ExperimentatorClient(EXPERIMENTS_PATH)
    client.load_experiment("exp_car_plate_text")
    pipeline = client.get_pipeline()

    # process images
    images  = os.listdir(input_path)
    results = []
    for image in tqdm(images):
        results.append(pipeline.predict(image))

    # combine to MVP dataset
    # todo

if __name__ == "__main__":
    main()
