import torch.nn
import torch.optim
import pytorch_lightning
import torchmetrics

class ExampleModel(pytorch_lightning.LightningModule):
    NUM_CLASSES = 10
    def __init__(self, lr=1e-5, my_hyperparam=1.5):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Flatten(),
            torch.nn.Linear(200, self.NUM_CLASSES)
        )

    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, data, *args, **kwargs):
        loss_fn = torch.nn.CrossEntropyLoss()
        inputs, targets = data
        predictions = self(inputs)
        loss = loss_fn(predictions, targets)
        self.log("train/loss", loss)
        return loss

    def on_validation_epoch_start(self) -> None:
        self.acc = torchmetrics.Accuracy("multiclass", num_classes=self.NUM_CLASSES).to(self.device)

    def validation_step(self, data, *args, **kwargs):
        entropy_fn = torch.nn.CrossEntropyLoss()
        inputs, targets = data
        prediction = self(inputs)
        loss = entropy_fn(prediction, targets)
        self.log("val/loss", loss)

        self.acc.update(prediction, targets)

    def on_validation_epoch_end(self):
        self.log("val/acc", self.acc.compute())

    def test_step():
        pass
