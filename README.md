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
tensorboard --bind_all --logdir ./logs --port 6006
```
and open it in any browser on address ```http://localhost:6006/```.
