
Train an extremely small diffusion model for tiny images (16x16)

## Train a model

```
python3 train.py -data_path "data/Apple/*"   -save_path ckpt/emoji
```

## Sample

```
python3 sample.py -model ckpt/emoji -num 100 -output out.png -noise_mul 10
```


## Acknowledge

This work is inspired from [pixelart-diffusion](https://github.com/zzbuzzard/pixartdiffusion). We also forked and reorganized some code from this repo.