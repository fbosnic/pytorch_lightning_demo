## A minimal example of lightning project
Minimal [lightning](https://pytorch-lightning.readthedocs.io/en/stable/) (previosly Pytorch Lightning) project to get started.
Lightning main repo


To run, install conda envirnoment with

```
conda env create --file requirements.yml
```

and then start the training with
```
python train.py
```

Tensorboard logs will be available in ```./logs```. You can run tensorboard with
```
tensorboard --bind-all ./logs --port 6006
```
