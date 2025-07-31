# Accelerating SGDM via Learning Rate and Batch Size Schedules: A Lyapunov-Based Analysis
Source code for reproducing our paper's experiments.

# Abstract
We analyze the convergence behavior of stochastic gradient descent with momentum (SGDM) under dynamic learning rate and batch size schedules by introducing a novel Lyapunov function. This Lyapunov function has a simpler structure compared with existing ones, facilitating the challenging convergence analysis of SGDM and a unified analysis across various dynamic schedules. Specifically, we extend the theoretical framework to cover three practical scheduling strategies commonly used in deep learning: (i) constant batch size with a decaying learning rate, (ii) increasing batch size with a decaying learning rate, and (iii) increasing batch size with an increasing learning rate. Our theoretical results reveal a clear hierarchy in convergence behavior: while (i) does not guarantee convergence of the expected gradient norm, both (ii) and (iii) do. Moreover, (iii) achieves a provably faster decay rate than (i) and (ii), demonstrating theoretical acceleration even in the presence of momentum. Empirical results validate our theory, showing that dynamically scheduled SGDM significantly outperforms fixed-hyperparameter baselines in convergence speed. We also evaluated a warm-up schedule in experiments, which empirically outperformed all other strategies in convergence behavior. These findings provide a unified theoretical foundation and practical guidance for designing efficient and stable training procedures in modern deep learning.

# Wandb Setup
Please change entity name XXXXXX to your wandb entitiy.
```bash
wandb.init(config=config, project=wandb_project_name, name=wandb_exp_name, entity="XXXXXX")
```

# Usage
```bash
python cifar100.py XXXXXX.json
```

# Example JSON Configuration
```bash
{
    "model": "resnet18",
    "bs_method": "constant",
    "lr_method": "constant",
    "beta_method": "constant",
    "init_bs": 128,
    "init_lr": 0.1,
    "init_beta": 0.9,
    "epochs": 300,
    "nshb": false,
    "use_wandb": true
}
```
| Parameter | Value | Description |
| :-------- | :---- | :---------- |
| `model` | `"resnet18"`, `"WideResNet28_10"`, etc. | Specifies the model architecture to use. |
| `bs_method` | `"constant"`, `"exp_growth"` | Method for adjusting the batch size. |
|`lr_method`|`"constant"`, `"cosine"`, `"diminishing"`, `"linear"`, `"poly"`, <br>`"exp_growth"`,`"warmup_const"`, `"warmup_cosine"`|Method for adjusting the learning rate.|
|`beta_method`|`"constant"`|Method for adjusting the momentum parameter $\beta$. <br> In this study, only "constant" is used.|
|`init_bs`|`int` (e.g., `128`)| The initial batch size for the optimizer. |
|`bs_max`|`int` (e.g., `4096`)| The maximum batch size to be reached when increasing the batch size, if an upper limit is desired. Used when `bs_method` is `"exp_growth"`.|
|`init_lr`|`float` (e.g., `0.1`)| The initial learning rate for the optimizer. |
|`lr_max`|`float` (e.g., `0.2`)|The maximum learning rate to be reached when increasing the learning rate, if an upper limit is desired. Used when `lr_method` is `"exp_growth"`, `"warmup_const"`, or `"warmup_cosine"`.|
|`lr_min`|`float` (e.g., `0.001`)| The minimum learning rate to be used in the cosine annealing schedule. Used when `lr_method` is `"cosine"` or `"warmup_cosine"`. The default value is `0`.|
|`epochs`|`int` (e.g., `300`)|The total number of epochs for training.|
|`incr_interval`|`int` (e.g., `30`)|Interval (in epochs) at which the batch size will increase. Also, the interval for increasing the learning rate when `lr_method` is `"exp_growth"`. Used when `bs_method` is `"exp_growth"`.|
|`warmup_epochs`|`int` (e.g., `30`)|Number of epochs over which the learning rate warms up from `init_lr` to `lr_max`. Used when `lr_method` is `"warmup_const"` or `"warmup_cosine"`.|
|`warmup_interval`|`int` (e.g., `3`)|The interval (in epochs) during which the learning rate increases in the warmup phase. Used when `lr_method` is `"warmup_const"` or `"warmup_cosine"`.|
|`bs_growth_rate`|`float` (e.g., `2.0`)|The factor by which the batch size increases after each interval. Used when `bs_method` is `"exp_growth"`.|
|`lr_growth_rate`| `float` (e.g., `1.2`) |The factor by which the learning rate increases after each interval. Used when `lr_method` is `"exp_growth"`, `"warnup_const`", or `"warmup_cosine"`.|
|`power`| `float` (e.g., `2.0`) |A parameter used when `lr_method` is set to `"poly"`, defining the polynomial decay rate of the learning rate.|
|`nshb| `boolean` (e.g., `true`) | Selects the momentum update rule. Set to `false` to use SHB: $m_t = \beta m_{t-1} + \nabla f(\theta_{t})$, and `true` to use NSHB: $m_t = \beta m_{t-1} + (1-\beta) \nabla f(\theta_{t})$. |
