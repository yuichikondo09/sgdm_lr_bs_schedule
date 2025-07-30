import math
from typing import Optional, Callable

class BetaScheduler:

    def __init__(self, initial_beta: int, last_step: int = -1):
        self.initial_beta = initial_beta
        self.last_step = last_step
        self.beta = initial_beta

        self._initial_step()

    def _initial_step(self):
        """Perform initial step and update batch size"""
        self.step()

    def get_beta(self):
        # Compute batch size using chainable form of the scheduler
        raise NotImplementedError

    def step(self, steps: Optional[int] = None):
        if steps is None:
            self.last_step += 1
        else:
            self.last_step = steps

        self.beta = self.get_beta()

    def state_dict(self):
        """Returns the state of the scheduler as a dictionary."""
        return {
            'initial_beta': self.initial_beta,
            'last_step': self.last_step,
            'beta': self.beta
        }

    def load_state_dict(self, state_dict):
        """Loads the scheduler's state from a state_dict."""
        self.initial_beta = state_dict['initial_beta']
        self.last_step = state_dict['last_step']
        self.beta = state_dict['beta']


class LambdaBeta(BetaScheduler):
    """Sets the batch size of each parameter group to the initial bs
    times a given function. When last_epoch=-1, sets initial bs as bs.

    Args:
        initial_bs (int): Initial batch size.
        bs_lambda (Callable[[int], int]): A function that takes the current epoch
            as an input and returns the factor to adjust the batch size.
        last_epoch (int): The index of last epoch. Default: -1.
    """
    def __init__(
        self,
        initial_beta,
        beta_lambda: Callable[[int], float],
        last_step: int = -1
    ):
        self.beta_lambda = beta_lambda
        super().__init__(initial_beta, last_step)

    def get_beta(self):
        return self.beta_lambda(self.last_step)
