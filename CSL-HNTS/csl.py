

import os
import torch
import logging
from base import BaseLearner,Tensor
from trainers.al_trainer import ALTrainer
from models.model import AutoEncoder
import threshold


def set_seed(seed):
    torch.manual_seed(seed)
    try:
        os.environ['PYTHONHASHSEED'] = str(seed)
    except Exception:
        pass


class CSL(BaseLearner):
    def __init__(self,
                 input_dim=1,
                 hidden_layers=1,
                 hidden_dim=4,
                 activation=torch.nn.LeakyReLU(0.05),
                 epochs=50,
                 update_freq=10,
                 timestep=10,
                 init_iter=10,
                 lr=1e-3,
                 cuasal_sparsity=0.3,
                 early_stopping=True,
                 early_stopping_thresh=1.0,
                 seed=47,  #6 20 40 43
                 device_type='cpu',
                 device_ids='0'):

        super(CSL, self).__init__()
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.epochs = epochs
        self.update_freq = update_freq
        self.init_iter = init_iter
        self.timestep=timestep
        self.lr = lr
        self.cuasal_sparsity = cuasal_sparsity
        self.early_stopping = early_stopping
        self.early_stopping_thresh = early_stopping_thresh
        self.seed = seed
        self.device_type = device_type
        self.device_ids = device_ids

        if torch.cuda.is_available():
            logging.info('GPU is available.')
        else:
            logging.info('GPU is unavailable.')
            if self.device_type == 'gpu':
                raise ValueError("GPU is unavailable, "
                                 "please set device_type = 'cpu'.")
        if self.device_type == 'gpu':
            if self.device_ids:
                os.environ['CUDA_VISIBLE_DEVICES'] = str(self.device_ids)
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        self.device = device

    def learn(self, data, columns=None, **kwargs):

        x = torch.from_numpy(data)

        self.n, self.d = x.shape[:2]
        if x.ndim == 2:
            x = x.reshape((self.n, self.d, 1))
            self.input_dim = 1
        elif x.ndim == 3:
            self.input_dim = x.shape[2]

        change = self._gae(x).detach().cpu().numpy()

        self.weight_causal_matrix = Tensor(change,
                                           index=columns,
                                           columns=columns)
        thr=threshold.auto_thre(self.cuasal_sparsity,torch.from_numpy(abs(change)))
        causal_matrix = (abs(change) > thr).astype(int)   #看绝对值

        self.causal_matrix = Tensor(causal_matrix,
                                    index=columns,
                                    columns=columns)

    def _gae(self, x):

        set_seed(self.seed)
        model = AutoEncoder(d=self.d,
                            input_dim=self.input_dim,
                            hidden_layers=self.hidden_layers,
                            hidden_dim=self.hidden_dim,
                            activation=self.activation,
                            timestep=self.timestep,
                            device=self.device,
                            )
        trainer = ALTrainer(n=self.n,
                            d=self.d,
                            model=model,
                            lr=self.lr,
                            init_iter=self.init_iter,
                            early_stopping=self.early_stopping,
                            early_stopping_thresh=self.early_stopping_thresh,
                            seed=self.seed,
                            device=self.device)
        change = trainer.train(x=x,
                              epochs=self.epochs,
                              update_freq=self.update_freq)
        change = change / torch.max(abs(change))

        return change
