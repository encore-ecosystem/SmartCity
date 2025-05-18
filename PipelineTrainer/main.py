from experimentator import ExperimentatorClient, Experiment, Pipeline, Trainer, Measurer
from src import EXPERIMENTATOR_PATH, DATASETS_PATH
from prologger import ConsoleLogger
from cvtk import MVP_Dataset
from platescanner.models import Yolo, Plug

USE_JOIN_SEQUENCE = True
logger = ConsoleLogger()
client = ExperimentatorClient(EXPERIMENTATOR_PATH)


def get_datasets() -> list[MVP_Dataset]:
    dataset = MVP_Dataset.read(
        DATASETS_PATH / "PlateDetector"
    )
    return [dataset for _ in range(9)]

def train_join():
    datasets = get_datasets()

    joined_experiment = Experiment(
        name    = "full_sequence",
        modules = [
            (
                Yolo(
                ),
                Trainer(
                    train_dataset   = datasets[0],
                    test_dataset    = datasets[1],
                    eval_dataset    = datasets[2],
                    epochs          = 10,
                    checkpoint_step = 2,
                    resume          = True,
                    measurer        = Measurer(),
                    logger          = logger
                ),
            ),
            (
                Yolo(
                ),
                Trainer(
                    train_dataset   = datasets[3],
                    test_dataset    = datasets[4],
                    eval_dataset    = datasets[5],
                    epochs          = 10,
                    checkpoint_step = 2,
                    resume          = True,
                    measurer        = Measurer(),
                    logger          = logger
                ),
            ),
            (
                Plug(
                ),
                Trainer(
                    train_dataset   = datasets[6],
                    test_dataset    = datasets[7],
                    eval_dataset    = datasets[8],
                    epochs          = 10,
                    checkpoint_step = 2,
                    resume          = True,
                    measurer        = Measurer(),
                    logger          = logger
                ),
            ),
        ],
        logger  = logger,
    )
    client.add_experiment(joined_experiment)
    run()

def train_separate():
    # 1. Configure car detector
    car_detector_experiment = Experiment(
        name = "car_detector",
        pipeline = Pipeline(
            models = [

            ],
        ),
        trainers = [],
        logger = logger,
    )
    client.add_experiment(car_detector_experiment)

    # 2. Configure plate detector
    plate_detector_experiment = Experiment(
        name = "plate_detector",
        pipeline = Pipeline(
            models = [

            ],
        ),
        trainers = [],
        logger = logger,
    )
    client.add_experiment(plate_detector_experiment)

    # 3. Configure plate recognizer
    plate_recognizer_experiment = Experiment(
        name = "plate_recognizer",
        pipeline = Pipeline(
            models = [

            ],
        ),
        trainers = [],
        logger = logger,
    )
    client.add_experiment(plate_recognizer_experiment)
    run()

def run():
    client.run_experiments()


if __name__ == "__main__":
    train_join() if USE_JOIN_SEQUENCE else train_separate()

