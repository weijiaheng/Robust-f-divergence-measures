# When Optimizing f-Divergence is Robust with Label noise


## Required Packages & Environment
**Supported OS:** Windows, Linux, Mac OS X; Python: 3.6/3.7; 
**Deep Learning Library:** PyTorch (GPU required)
**Required Packages:** Numpy, Pandas, random, matplotlib, tqdm, csv, torch.



## Training -- CIothing 1M
We select around 250000 images so that all classes are balanced. Selected idx file is in `C1M_selected_idx_balance.pkl`. To run on Clothing 1M, readers have to download the dataset and put the data into corresponding directory. Then, simply run:

```
python3 runner_C1M.py --r noise --s seed --divergence 'name' --warmup 0 --batchsize 128
```

Hints:
> 📋 (1) For Clothing 1M real-world noisy dataset, we recommend training without bias correction (no need to install Keras, Tensorflow, Pytorch in your virtual environment) and load a pre-trained ResNet50 Model;

> 📋 (2) Suggested divergence functions: Total-Variation, Jenson-Shannon, KL, Pearson, Jeffrey.
