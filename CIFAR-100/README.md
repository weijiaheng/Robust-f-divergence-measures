# When Optimizing f-Divergence is Robust with Label noise

## Required Packages & Environment
**Supported OS:** Windows, Linux, Mac OS X; Python: 3.6/3.7; 
**Deep Learning Library:** PyTorch (GPU required)
**Required Packages:** Numpy, Pandas, random, matplotlib, tqdm, csv, torch.


## Training -- CIFAR-100
> 📋Step 1:
Train CE model as a warm-up:
```
python3 runner_CE.py --r noise --s seed --divergence 'name' --warmup 0 --batchsize 128
```
Sparse-Low | Sparse-High | Uniform-Low | Uniform-High 
--- | --- | --- | --- 
r=0.1 | r=0.2 | r=0.3 | r=0.4

Used divergence list:  Total-Variation, Jenson-Shannon, Pearson, KL, Jeffrey.

> 📋Step 2: Maximizing $D_f$ measures
### Without bias correction:
```
python3 runner.py --r noise --s seed --divergence 'name' --warmup 0 --batchsize 128
```


Hints:
> 📋 (1) For CIFAR-100 noisy dataset, we recommend training without bias correction (no need to install Keras, Tensorflow, Pytorch in your virtual environment);

> 📋 (2) The training is stable if: warm-up with CE loss or load a pre-trained CE model;

> 📋 (3) Suggested divergence functions: Total-Variation, Jenson-Shannon, KL, Pearson, Jeffrey.
