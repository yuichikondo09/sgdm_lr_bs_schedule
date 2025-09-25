import torch
import torch.optim as optim
from utils.get_config_value import get_config_value

class SGDM(optim.Optimizer):
    def __init__(self, params, lr, config):
        defaults = dict(lr=lr, config=config)
        super(SGDM, self).__init__(params, defaults)
        
        self.beta = get_config_value(config, "init_beta")
        self.nshb = get_config_value(config, "optimizer") == "nshb"

        for group in self.param_groups:
            for p in group['params']:
                self.state[p] = {'momentum': torch.zeros_like(p.data)}
            
        if self.nshb:
            self._update_momentum = lambda prev_m, grad: prev_m.mul(self.beta).add_(grad, alpha=(1.0 - self.beta))
        else:
            self._update_momentum = lambda prev_m, grad: prev_m.mul(self.beta).add_(grad)

    def step(self, iteration=0):

        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]
                prev_m = state['momentum']

                m_t = self._update_momentum(prev_m, grad)

                p.data.add_(m_t, alpha=-lr)

                state['momentum'] = m_t.clone()

        if iteration != 0:
            gradient_list = []
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        gradient_list.append(p.grad.data.clone())
            return gradient_list, momentum_list
        return None