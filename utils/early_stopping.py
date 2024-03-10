import os
import numpy as np
import torch


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience, verbose, delta, mode, model, output_path):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            mode (str): Procedure for determining the best score.
            model : Will be used to store best models
            output_path (str): Path where models are saved


        """

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.early_stop = False
        self.mode = mode
        self.model = model
        self.output_path = output_path

        if self.mode == 'min':
            self.criterion = np.less
            self.delta = - delta
            self.best_score = np.Inf

            self.vocab = {'score': 'loss', 'comportement': 'decreased'}

        elif self.mode == 'max':
            self.criterion = np.greater
            self.delta = delta
            self.best_score = np.NINF

            self.vocab = {'score': 'metric', 'comportement': 'increased'}

        else:
            raise ValueError(
                "mode only takes as value in input 'min' or 'max'")

    def __call__(self, score, model):
        """Determines if the score is the best and saves the model if so.
           Also manages early stopping.
        Arguments:
            score (float): Value of the metric or loss.
            model: Pytorch model
        """
        if np.isinf(self.best_score):
            self.best_score = score
            self.save_checkpoint(score, model)

        elif self.criterion(score, self.best_score + self.delta):

            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0
            self.model = model
        else:
            self.counter += 1
            print(
                f'EarlyStopping counter: {self.counter} out of {self.patience}'
            )
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, score, model):
        '''Saves the model when the score satisfies the criterion.'''
        if self.verbose:
            score_name = self.vocab['score']
            comportement = self.vocab['comportement']
            print(
                f'Validation {score_name} {comportement} ({self.best_score:.6f} --> {score:.6f}).  Saving model ...'
            )
        model_path = os.path.join(self.output_path, 'best_model.pth')
        torch.save(model, model_path)

    def get_best_score(self):
        return self.best_score
