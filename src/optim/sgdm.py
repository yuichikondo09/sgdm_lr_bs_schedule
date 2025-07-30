import torch
import torch.optim as optim
from utils.get_beta_scheduler import get_beta_scheduler
from utils.get_config_value import get_config_value

class SGDM(optim.Optimizer):
    def __init__(self, params, lr, config, total_steps):
        defaults = dict(lr=lr, config=config)
        super(SGDM, self).__init__(params, defaults)
        
        self.beta_init = get_config_value(config, "init_beta")
        self.beta_schedule, self.beta_step_type = get_beta_scheduler(config, total_steps)
        self.nshb = get_config_value(config, "nshb")

        for group in self.param_groups:
            for p in group['params']:
                self.state[p] = {'momentum': torch.zeros_like(p.data)}
            
        if self.nshb:
            self._update_momentum = lambda prev_m, beta_t, grad: prev_m.mul(beta_t).add_(grad, alpha=(1.0 - beta_t))
        else:
            self._update_momentum = lambda prev_m, beta_t, grad: prev_m.mul(beta_t).add_(grad)

    def step(self, iteration=0):
        beta_t = self.beta_init * self.beta_schedule.get_beta()

        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]
                prev_m = state['momentum']

                m_t = self._update_momentum(prev_m, beta_t, grad)

                p.data.add_(m_t, alpha=-lr)

                state['momentum'] = m_t.clone()

        if self.beta_step_type == 'step':
            self.beta_schedule.step()

        if iteration != 0:
            gradient_list = []
            momentum_list = []
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        gradient_list.append(p.grad.data.clone())
                        momentum_list.append(self.state[p]['momentum'].clone())
            return gradient_list, momentum_list
        return None