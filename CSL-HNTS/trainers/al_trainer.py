import numpy as np
import torch
import logging

from consts import LOG_FREQUENCY, LOG_FORMAT

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


class ALTrainer(object):

    def __init__(self, n, d, model, lr, init_iter, early_stopping,
                 early_stopping_thresh, seed, device=None):
        self.n = n
        self.d = d
        self.model = model
        self.lr = lr
        self.init_iter = init_iter
        self.early_stopping = early_stopping
        self.early_stopping_thresh = early_stopping_thresh
        self.seed = seed
        self.device = device

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.lr)

    def train(self, x, epochs, update_freq):
        for epoch in range(epochs):
            logging.info(f'Current epoch: {epoch}==================')
            mse_new, h_new, w_new,change = self.train_step(x,
                                                        update_freq)

        return change


    def train_step(self, x, update_freq):

        curr_mse, curr_h, w_adj = None, None, None
        for _ in range(update_freq):
            torch.manual_seed(self.seed)
            curr_mse, w_adj,change = self.model(x)
            torch.autograd.set_detect_anomaly(True)
            loss=  (0.5 / self.n) *curr_mse
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if _ % LOG_FREQUENCY == 0:
                logging.info(f'Current loss in step {_}: {loss.detach()}')

        return curr_mse, curr_h, w_adj,change


