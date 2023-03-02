from pathlib import Path
from torchvision.datasets import FashionMNIST
import torch.utils.data
import pytorch_lightning
import numpy as np

class ExampleDataModule(pytorch_lightning.LightningDataModule):
    def __init__(self, data_root=Path(__file__).parent / "DATA") -> None:
        super().__init__()
        self.data_root = data_root

    def prepare_data(self) -> None:

        raw_dataset = FashionMNIST(root=self.data_root, train=True, download=True,
                                   transform=lambda im: np.expand_dims(np.array(im, dtype=np.float32), axis=0)
                                   )
        train_val_spilt_index = int(0.8 * len(raw_dataset))
        self.train_dataset = torch.utils.data.Subset(raw_dataset, range(train_val_spilt_index))
        self.val_dataset = torch.utils.data.Subset(raw_dataset, range(train_val_spilt_index, len(raw_dataset)))
        self.test_dataset = FashionMNIST(root=self.data_root, train=False, download=True)

    def setup(self, stage) -> None:
        pass

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=128)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=16)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=2)

    def predict_dataloader(self):
        return None
