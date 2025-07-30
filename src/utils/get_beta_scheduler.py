import math
from .get_config_value import get_config_value
from optim.beta_scheduler import LambdaBeta

def diminishing_beta_lambda(steps):
    return 1 / math.sqrt(steps + 1)

def linear_beta_lambda(steps, total_steps):
    return 1 - (steps / total_steps)

def cosine_beta_lambda(steps, beta_init, T_max, beta_min):
    return beta_min + (beta_init-beta_min)*(1+math.cos((steps*math.pi)/T_max))/2.

def demon_beta_lambda(steps, beta_init, total_steps):
    return (total_steps - steps) / (total_steps - beta_init*steps)

def increasing_beta_lambda(steps, beta_init):
    return 1-(1-beta_init)/math.sqrt(steps+1)

def cosine_growth_beta_lambda(steps, beta_init, T_max, beta_max):
    return beta_init + (beta_max-beta_init)*(1-math.cos((steps*math.pi)/T_max))/2.

def linear_growth_beta_lambda(steps, beta_init, total_steps):
    return beta_init + (1-beta_init)*steps/total_steps

def demon_growth_beta_lambda(steps, beta_init, total_steps):
    return 1- (1-beta_init)*(total_steps - steps) / (total_steps - (1-beta_init)*steps)

def step_incr_beta_lambda(epoch):
    if epoch <= 100:
        return 0.9
    elif epoch <= 200:
        return 0.99
    else:
        return 0.999


def get_beta_scheduler(config, total_steps):
    method = get_config_value(config, "beta_method")
    beta_init = get_config_value(config, "init_beta")

    if method == "constant":
        f = lambda epoch: beta_init
        step_type = "epoch"
    elif method == "diminishing":
        f = lambda step: beta_init*diminishing_beta_lambda(step)
        step_type = "step"
    elif method == "cosine":
        f = lambda epoch: cosine_beta_lambda(epoch, beta_init, T_max=get_config_value(config, "epochs"), beta_min=config.get("beta_min", 0))
        step_type = "epoch"
    elif method == "linear":
        f = lambda step: beta_init*linear_beta_lambda(step, total_steps)
        step_type = "step"
    elif method == "demon":
        f = lambda step: beta_init*demon_beta_lambda(step, beta_init, total_steps)
        step_type = "step"

    elif method == "increasing":
        f = lambda step: increasing_beta_lambda(step, beta_init)
        step_type = "step"
    elif method == "cosine_growth":
        f = lambda epoch: cosine_growth_beta_lambda(epoch, beta_init, T_max=get_config_value(config, "epochs"), beta_max=config.get("beta_max", 0.99))
        step_type = "epoch"
    elif method == "linear_growth":
        f = lambda step: linear_growth_beta_lambda(step, beta_init, total_steps)
        step_type = "step"
    elif method == "demon_growth":
        f = lambda step: demon_growth_beta_lambda(step, beta_init, total_steps)
        step_type = "step"
    elif method == "step_incr":
        f = lambda epoch: step_incr_beta_lambda(epoch)
        step_type = "epoch"
    else:
        raise ValueError(f"Unknown beta_method: {method!r}")
    
    scheduler = LambdaBeta(beta_init, beta_lambda=f)
    return scheduler, step_type