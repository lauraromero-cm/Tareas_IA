# models.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset


class GameDataset(Dataset):
    def __init__(self, X, y):
        # X, y vienen como numpy arrays
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LogisticRegressionTorch(nn.Module):
    """
    Regresión logística multiclase:
    logits = Wx + b
    CrossEntropyLoss aplica softmax internamente.
    """
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.linear(x)


class LinearSVMTorch(nn.Module):
    """
    Clasificador lineal multiclase con margen tipo hinge.
    """
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.linear(x)

    def multiclass_hinge_loss(self, logits, y, margin=1.0):
        """
        Pérdida hinge multiclase estilo Crammer-Singer:
        sum(max(0, margin + score_j - score_ytrue)) para j != ytrue
        """
        batch_size = logits.shape[0]
        correct_scores = logits[torch.arange(batch_size), y].unsqueeze(1)
        margins = margin + logits - correct_scores
        margins[torch.arange(batch_size), y] = 0.0
        loss = torch.clamp(margins, min=0.0).mean()
        return loss

