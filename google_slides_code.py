import pytorch_lightning

class MyDataModule(pytorch_lightning.LightningDataModule):
    def setup(self, stage: str): pass
    def train_dataloader(self): pass

class MyModel(pytorch_lightning.LightningModule):
    def forward(self, x): pass
    def training_step(self, batch, batch_idx): pass
    def configure_optimizers(self): pass

datamodule = MyDataModule()
model = MyModel()
trainer = pytorch_lightning.Trainer()
trainer.fit(model, datamodule)



from pytorch_lightning import Trainer

# 8 GPUs, no other code changes needed
trainer = Trainer(gpus=8)

# 256 GPUs, 32 nodes
trainer = Trainer(gpus=8, num_nodes=32)

# tpu training
trainer = Trainer(tpu_cores=8)

# torch.float16
trainer = Trainer(precision=16)



from pytorch_lightning import loggers

logger = loggers.TensorBoardLogger('logs/')
logger = loggers.CometLogger()
logger = loggers.MLFlowLogger()
logger = loggers.CSVLogger()
logger = loggers.WandbLogger()
logger = loggers.NeptuneLogger()
# ... and others

trainer = Trainer(logger=logger)


from pytorch_lightning import callbacks

early_stopping = callbacks.EarlyStopping(
        monitor="val/loss"
    )
checkpoint_callback = callbacks.ModelCheckpoint(
        monitor="val/loss", save_top_k=2
    )
trainer = Trainer(
        callbacks=[early_stopping, checkpoint_callback]
    )
