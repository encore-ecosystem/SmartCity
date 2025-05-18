from cvtk import MVP_Dataset
from platescanner.pipelines import PTPipeline
from pathlib import Path
from tqdm import tqdm
import os
import shutil
import numpy as np


ROOT_PATH = Path().resolve()
RENDERS_PATH = ROOT_PATH.parent / "SmartCity" / "Saved" / "MovieRenders"
DATASETS_PATH = ROOT_PATH / "Datasets"

def main():
    # guards
    if not RENDERS_PATH.exists():
        raise ValueError("Path does not exists")
    if not RENDERS_PATH.is_dir():
        raise ValueError("Path should be directory")

    # make / load experimentator pipeline
    pipeline = PTPipeline()

    # process images
    pbar = tqdm(os.listdir(RENDERS_PATH))
    for sequence_name in pbar:
        pbar.set_description(f"Predicting sequence: {sequence_name}")

        images  = os.listdir(RENDERS_PATH / sequence_name)
        results = []
        for image in tqdm(images):
            results.append(pipeline.predict_on_image(RENDERS_PATH / sequence_name / image))

        # Create MVP Dataset
        ttv_weights = np.array([3., 2., 1.])
        ttv_weights /= np.sum(ttv_weights)
        n = len(images)
        p1 = int(n * ttv_weights[0])
        p2 = p1 + int(n * ttv_weights[1])

        fetched = 0
        splits = {
            "train" : images[:p1],
            "test"  : images[p1:p2],
            "valid" : images[p2:],
        }
        
        mvp_path = DATASETS_PATH / sequence_name
        if mvp_path.exists():
            shutil.rmtree(mvp_path)
        mvp_path.mkdir(parents=True)

        mvp_images = {}
        mvp_attributes = {}
        for split in splits:
           mvp_images[split] = {}
           mvp_attributes[split] = {}
           for image in splits[split]:
                mvp_images[split][image] = RENDERS_PATH / sequence_name / image
                mvp_attributes[split][f"{mvp_images[split][image].stem}.txt"] = {
                    'Detection': {
                        'bboxes': [
                            {
                                'superclasses'     : [],
                                'bbox_type'        : "bb",
                                'class_name'       : "plate",
                                'points'           : bbox.bbox,
                                'recognition_text' : bbox.value,

                            } for bbox in results[fetched]
                        ],
                        'keypoints': [],
                    },

                    'Segmentation': {
                        'polygons': [
                            [],
                        ],
                    }
                }
                fetched += 1

        MVP_Dataset(
            path = mvp_path,
            manifest = {},
            images = mvp_images,
            attributes= mvp_attributes,
        ).write(mvp_path)


if __name__ == "__main__":
    main()
