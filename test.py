from datamodule import ExampleDataModule

dm = ExampleDataModule()
dm.prepare_data()
dm.setup()

dl = dm.train_dataloader()
print(next(iter(dl))[0].shape)