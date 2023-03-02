from pathlib import Path
from datamodule import ExampleDataModule
from model import ExampleModel
import pytorch_lightning as lightning
import pytorch_lightning.loggers as loggers
import pytorch_lightning.callbacks as callbacks
import random


if __name__ == "__main__":
    root = Path(__file__).parent
    model = ExampleModel(my_hyperparam=random.uniform(0, 1))
    datamodule = ExampleDataModule(root / "DATA")
    tensorboard_logger = loggers.TensorBoardLogger(
        root / "logs",
        name="my_experiment",
        default_hp_metric=False
    )

    trainer = lightning.Trainer(
        logger=tensorboard_logger,
        accelerator="gpu",
        devices=1,
        callbacks=[
            callbacks.EarlyStopping(monitor="val/loss"),
            callbacks.ModelCheckpoint(monitor="val/loss", save_top_k=2)
        ]
    )
    trainer.fit(model, datamodule=datamodule)
