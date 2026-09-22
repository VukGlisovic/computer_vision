# Super Resolution based Compression


This project is based on the paper "Lossless Image Compression through Super-Resolution" (https://arxiv.org/pdf/2004.02872). The code is based on https://github.com/caoscott/SReC.

The idea behind this paper, is to train a model that can basically predict the next pixel based on neighbouring pixels. By using such a trained model, we can compress images significantly while preserving the exact pixel values. It is basically the same as storing a file in PNG format (which is also a lossless compression algorithm), except that it should be even more effecient and thus the file size should be smaller. The downside of course, is that you need the model to decode the compressed image again.


## Training

```bash
pixi run train        # settings come from scripts/config.yaml
pixi run tensorboard  # metrics at http://localhost:6006
```

Checkpoints are written to `experiments/lightning_logs/version_*/checkpoints/`.


## Compressing

Both tasks run from `scripts/`, so paths are relative to that directory.

```bash
CKPT=../experiments/lightning_logs/version_0/checkpoints/model_epoch=25.ckpt

pixi run compress -c $CKPT -i ../resources/cifar10_ship.png                 # writes cifar10_ship.srec
pixi run compress -c $CKPT -i ../resources/cifar10_ship.srec -o restored.png
```

Whether a file is compressed or restored follows from its contents, not from its name. Compressing also
decodes the result again and fails if the round trip is not lossless. A stream can only be read back with
the checkpoint that wrote it.


## Results

The three 32x32 CIFAR-10 images in `resources`, compressed with `model_epoch=25.ckpt`. The PNG column is
the same pixels re-encoded with `optimize=True`, which is a slightly stronger baseline than the files on disk.

| image | png | srec | saving |
| --- | --- | --- | --- |
| `cifar10_airplane.png` | 2149 B (5.5964 bpsp) | 1767 B (4.6016 bpsp) | 17.8% |
| `cifar10_cat.png` | 2492 B (6.4896 bpsp) | 1683 B (4.3828 bpsp) | 32.5% |
| `cifar10_ship.png` | 2149 B (5.5964 bpsp) | 1419 B (3.6953 bpsp) | 34.0% |
