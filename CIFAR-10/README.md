# When Optimizing f-Divergence is Robust with Label noise

This repository is the official implementation of "[When Optimizing f-Divergence is Robust with Label noise](https://arxiv.org/abs/2011.03687)" accepted by ICLR2021. 


## Required Packages & Environment
**Supported OS:** Windows, Linux, Mac OS X; Python: 3.6/3.7; 
**Deep Learning Library:** PyTorch (GPU required)
**Required Packages:** Numpy, Pandas, random, matplotlib, tqdm, csv, torch.





## Training -- CIFAR-10
> 📋Step 1:
Train CE model as a warm-up:
```
python3 runner_CE.py --r noise --s seed --divergence 'name' --warmup 0 --batchsize 128
```
Sparse-Low | Sparse-High | Uniform-Low | Uniform-High | Random-Low | Random-High 
--- | --- | --- | --- |--- |---
r=0.1 | r=0.2 | r=0.3 | r=0.4 | r=0.5 | r=0.6

Used divergence list:  Total-Variation, Jenson-Shannon, Pearson, KL, Jeffrey.

> 📋Step 2: Maximizing D_f measures
### Without bias correction:
```
python3 runner.py --r noise --s seed --divergence 'name' --warmup 0 --batchsize 128
```

### With bias correction:
3 choices are listed for bias correction:
(1) Having access to the noise rates:
Use the clean dataset to estimate the noise transition matrix (implemented);

(2) Use existed estimation methods:
In the folder named `Estimate`, we modifed code from Loss Correction: `https://github.com/giorgiop/loss-correction.`  To get the noise transition matrix, first modify parameters and run
```
python3 CE_warmup.py
```
Then, run
```
python3 Estimate_noise.py
```
(3) Manually input a noise transition matrix.

```Remark: both our experiment results and theoretical analysis demonstrate the negligible effect of bias term, especially for abovee mentioned 5 divergences. 
```
Now suppose you have the transition matrix, for uniform noise model, run:
```
python3 runner_bias_uniform.py --r noise --s seed --divergence 'name' --warmup 0 --scale False --batchsize 128
```
For sparse noise model, run:
```
python3 runner_bias_sparse.py --r noise --s seed --divergence 'name' --warmup 0 --batchsize 128
```

> 📋More details and hyperparameter settings can be seen in the supplementary materials and the corresponding runners.


