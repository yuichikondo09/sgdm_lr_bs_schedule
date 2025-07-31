# Accelerating SGDM via Learning Rate and Batch Size Schedules: A Lyapunov-Based Analysis
Source code for reproducing our paper's experiments.

## Abstract
We analyze the convergence behavior of stochastic gradient descent with momentum (SGDM) under dynamic learning rate and batch size schedules by introducing a novel Lyapunov function. This Lyapunov function has a simpler structure compared with existing ones, facilitating the challenging convergence analysis of SGDM and a unified analysis across various dynamic schedules. Specifically, we extend the theoretical framework to cover three practical scheduling strategies commonly used in deep learning: (i) constant batch size with a decaying learning rate, (ii) increasing batch size with a decaying learning rate, and (iii) increasing batch size with an increasing learning rate. Our theoretical results reveal a clear hierarchy in convergence behavior: while (i) does not guarantee convergence of the expected gradient norm, both (ii) and (iii) do. Moreover, (iii) achieves a provably faster decay rate than (i) and (ii), demonstrating theoretical acceleration even in the presence of momentum. Empirical results validate our theory, showing that dynamically scheduled SGDM significantly outperforms fixed-hyperparameter baselines in convergence speed. We also evaluated a warm-up schedule in experiments, which empirically outperformed all other strategies in convergence behavior. These findings provide a unified theoretical foundation and practical guidance for designing efficient and stable training procedures in modern deep learning.

## Wandb Setup
Please change entity name XXXXXX to your wandb entitiy.
```bash
wandb.init(config=config, project=wandb_project_name, name=wandb_exp_name, entity="XXXXXX")
```

## Usage
```bash
python cifar100.py XXXXXX.json
```

## Example JSON Configuration

The following is an example configuration for training ResNet18 using the NSHB optimizer.  
Batch size, learning rate, and momentum are all set to constant values, and training runs for 300 epochs.

```json
{
  "model": "resnet18",
  "bs_method": "constant",
  "lr_method": "constant",
  "beta_method": "constant",
  "init_bs": 128,
  "init_lr": 0.1,
  "init_beta": 0.9,
  "epochs": 300,
  "nshb": true,
  "use_wandb": true
}

Below is a detailed description of each configuration parameter used in the JSON example:
| Parameter         | Type & Example                                                                                                                        | Description                                                                                                                        |
| :---------------- | :------------------------------------------------------------------------------------------------------------------------------------ | :--------------------------------------------------------------------------------------------------------------------------------- |
| `model`           | `string` (`"resnet18"`, `"WideResNet28_10"`, etc.)                                                                                    | Specifies the model architecture                                                                                                   |
| `bs_method`       | `string` (`"constant"`, `"exp_growth"`)                                                                                               | Method for adjusting the batch size                                                                                                |
| `lr_method`       | `string` (`"constant"`, `"cosine"`, `"diminishing"`,<br>`"linear"`, `"poly"`, `"exp_growth"`,<br>`"warmup_const"`, `"warmup_cosine"`) | Method for adjusting the learning rate                                                                                             |
| `beta_method`     | `string` (`"constant"`)                                                                                                               | Method for adjusting the momentum parameter (β). *Only constant is used in this study*                                             |
| `init_bs`         | `int` (`128`)                                                                                                                         | Initial batch size                                                                                                                 |
| `bs_max`          | `int` (`4096`)                                                                                                                        | Maximum batch size when increasing batch size. Used when `bs_method="exp_growth"`                                                  |
| `init_lr`         | `float` (`0.1`)                                                                                                                       | Initial learning rate                                                                                                              |
| `lr_max`          | `float` (`0.2`)                                                                                                                       | Maximum learning rate when increasing learning rate. Used when `lr_method="exp_growth"`,<br>`"warmup_const"`, or `"warmup_cosine"` |
| `lr_min`          | `float` (`0.001`, default `0`)                                                                                                        | Minimum learning rate for cosine annealing. Used when `lr_method="cosine"` or `"warmup_cosine"`                                    |
| `epochs`          | `int` (`300`)                                                                                                                         | Total number of training epochs                                                                                                    |
| `incr_interval`   | `int` (`30`)                                                                                                                          | Interval (in epochs) for increasing batch size or learning rate. Used when `bs_method="exp_growth"`                                |
| `warmup_epochs`   | `int` (`30`)                                                                                                                          | Number of warmup epochs. Used when `lr_method="warmup_const"` or `"warmup_cosine"`                                                 |
| `warmup_interval` | `int` (`3`)                                                                                                                           | Interval (in epochs) for learning rate increase during warmup. Used when `lr_method="warmup_const"` or `"warmup_cosine"`           |
| `bs_growth_rate`  | `float` (`2.0`)                                                                                                                       | Batch size growth factor. Used when `bs_method="exp_growth"`                                                                       |
| `lr_growth_rate`  | `float` (`1.2`)                                                                                                                       | Learning rate growth factor. Used when `lr_method="exp_growth"`, `"warmup_const"`, or `"warmup_cosine"`                            |
| `power`           | `float` (`2.0`)                                                                                                                       | Polynomial decay power. Used when `lr_method="poly"`                                                                               |
| `nshb`            | `boolean` (`true`/`false`)                                                                                                            | Momentum update rule selector.<br>`false`: SHB<br>`true`: NSHB                                                                     |
| `use_wandb`       | `boolean` (`true`/`false`)                                                                                                            | Enables logging to Weights & Biases (wandb)                                                                                        |
