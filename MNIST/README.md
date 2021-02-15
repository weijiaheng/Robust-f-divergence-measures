# Experiments on MNIST 

## Required Packages & Environment
**Supported OS:** Windows, Linux, Mac OS X; Python: 3.6/3.7; 
**Deep Learning Library:** PyTorch (GPU required)
**Required Packages:** Numpy, Pandas, random, matplotlib, tqdm, csv, torch (also need Keras, Scikit-learn if use bias correction).



## Training -- MNIST
We give 6 noise Fashion dataset as mentioned in our paper. To run the experiment without bias correction:
```
python3 runner.py --r noise --s seed --divergence 'name' --warmup 0 --batchsize 128
```

Two defualt learning rate settings are listed in the file `runner.py`.


## Training with bias correction -- MNIST
Even though we do not recommend training with bias correction, as an example, we give two runners when readers are interested in optimizing f-divergence with bias correction.
> 📋 (1) For sparse noise type, to run the experiment with bias correction:
```
python3 runner_bias_sparse.py --r noise --s seed --divergence 'name' --warmup 0 --batchsize 128
```
> 📋 (2) For uniform noise type, to run the experiment with bias correction:
```
python3 runner_bias_uniform.py --r noise --s seed --divergence 'name' --warmup 0 --batchsize 128
```

Corresponding noise file is:
Sparse-Low | Sparse-High | Uniform-Low | Uniform-High | Random-Low | Random-High
--- | --- | --- | --- | --- | ---
r=0.1 | r=0.2 | r=0.3 | r=0.4 | r=0.5 | r=0.6

Hints:
> 📋 (1) For MNIST noisy dataset, we recommend training without bias correction (no need to install Keras, Tensorflow);

> 📋 (2) No need to warm-up with CE loss;

> 📋 (3) Suggested divergence functions: Total-Variation, Jenson-Shannon, KL, Pearson, Jeffrey.

