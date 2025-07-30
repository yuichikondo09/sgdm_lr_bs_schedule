# Accelerating SGDM via Learning Rate and Batch Size Schedules: A Lyapunov-Based Analysis
Source code for reproducing our paper's experiments.

# Abstract
We analyze the convergence behavior of stochastic gradient descent with momentum (SGDM) under dynamic learning rate and batch size schedules by introducing a novel Lyapunov function. This Lyapunov function has a simpler structure compared with existing ones, facilitating the challenging convergence analysis of SGDM and a unified analysis across various dynamic schedules. Specifically, we extend the theoretical framework to cover three practical scheduling strategies commonly used in deep learning: (i) constant batch size with a decaying learning rate, (ii) increasing batch size with a decaying learning rate, and (iii) increasing batch size with an increasing learning rate. Our theoretical results reveal a clear hierarchy in convergence behavior: while (i) does not guarantee convergence of the expected gradient norm, both (ii) and (iii) do. Moreover, (iii) achieves a provably faster decay rate than (i) and (ii), demonstrating theoretical acceleration even in the presence of momentum. Empirical results validate our theory, showing that dynamically scheduled SGDM significantly outperforms fixed-hyperparameter baselines in convergence speed. We also evaluated a warm-up schedule in experiments, which empirically outperformed all other strategies in convergence behavior. These findings provide a unified theoretical foundation and practical guidance for designing efficient and stable training procedures in modern deep learning.

# Wandb Setup
Please change entity name XXXXXX to your wandb entitiy.
```bash
wandb.init(config=config, project=wandb_project_name, name=wandb_exp_name, entity="XXXXXX")
```
