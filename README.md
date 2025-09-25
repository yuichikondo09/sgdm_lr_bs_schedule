# Accelerating SGDM via Learning Rate and Batch Size Schedules: A Lyapunov-Based Analysis
Source code for reproducing our paper's experiments.

## Abstract
We analyze the convergence behavior of stochastic gradient descent with momentum (SGDM) under dynamic learning-rate and batch-size schedules by introducing a novel and simpler Lyapunov function. We extend the existing theoretical framework to cover three practical scheduling strategies commonly used in deep learning: a constant batch size with a decaying learning rate, an increasing batch size with a decaying learning rate, and an increasing batch size with an increasing learning rate. Our results reveal a clear hierarchy in convergence: a constant batch size does not guarantee convergence of the expected gradient norm, whereas an increasing batch size does, and simultaneously increasing both the batch size and learning rate achieves a provably faster decay. Empirical results validate our theory, showing that dynamically scheduled SGDM significantly outperforms its fixed-hyperparameter counterpart in convergence speed. We also evaluated a warm-up schedule in experiments, which empirically outperformed all other strategies in convergence behavior.

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
  "optimizer": "nshb",
  "model": "resnet18",
  "bs_method": "constant",
  "lr_method": "constant",
  "init_bs": 128,
  "init_lr": 0.1,
  "init_beta": 0.9,
  "epochs": 300,
  "use_wandb": true
}
```

Below is a detailed description of each configuration parameter used in the JSON example:

| Parameter | Type & Example | Description |
| :- | :- | :- |
| `optimizer` | `str` (`"nshb"`, `"shb"`, "sgd"`, `"rmsprop"`, "adam"`, `"adamw"`) | Specifies the optimizer to use during training. |
| `model` | `str` (`"resnet18"`, `"resnet34"` etc.) | Specifies the model architecture |
| `bs_method` | `str` (`"constant"`, `"exp_growth"`) | Method for adjusting the batch size |
| `lr_method` | `str` (`"constant"`, `"cosine"`, `"diminishing"`,<br>`"linear"`, `"poly"`, `"exp_growth"`,<br>`"warmup_const"`, `"warmup_cosine"`) | Method for adjusting the learning rate |
| `init_bs` | `int` (`128`) | Initial batch size |
| `init_lr` | `float` (`0.1`) | Initial learning rate |
| `init_beta` | `float` (`0.9`) | Initial beta |
| `bs_max` | `int` (`4096`) | Maximum batch size when increasing batch size. Used when `bs_method="exp_growth"` |
| `lr_max` | `float` (`0.2`) | Maximum learning rate when increasing learning rate. Used when `lr_method="exp_growth"`,<br>`"warmup_const"`, or `"warmup_cosine"` |
| `lr_min` | `float` (`0.001`, default `0`) | Minimum learning rate for cosine annealing. Used when `lr_method="cosine"` or `"warmup_cosine"` |
| `epochs` | `int` (`300`) | Total number of training epochs |
| `incr_interval` | `int` (`30`) | Interval (in epochs) for increasing batch size or learning rate. Used when `bs_method="exp_growth"` |
| `warmup_epochs` | `int` (`30`) | Number of warmup epochs. Used when `lr_method="warmup_const"` or `"warmup_cosine"` |
| `warmup_interval` | `int` (`3`) | Interval (in epochs) for learning rate increase during warmup. Used when `lr_method="warmup_const"` or `"warmup_cosine"` |
| `bs_growth_rate` | `float` (`2.0`) | Batch size growth factor. Used when `bs_method="exp_growth"` |
| `lr_growth_rate` | `float` (`1.2`) | Learning rate growth factor. Used when `lr_method="exp_growth"`, `"warmup_const"`, or `"warmup_cosine"` |
| `power` | `float` (`2.0`) | Polynomial decay power. Used when `lr_method="poly"`|
| `use_wandb` | `boolean` (`true`/`false`) | Enables logging to Weights & Biases (wandb) |

