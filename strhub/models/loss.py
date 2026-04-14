import torch
import torch.nn as nn
from opt_einsum import contract

def compute_loss_mat(Z, y, eps=0.1):
    m, d = Z.shape
    I = torch.eye(d).to(Z.device)
    c = d / (m * eps)
    loss_expd = logdet(c * covariance(Z) + I) / 2.
    loss_comp = 0.

    unique_labels, counts = torch.unique(y, return_counts=True)
    for j, count in zip(unique_labels, counts):
        indices = (y == j).nonzero(as_tuple=True)[0]
        Z_j = Z[indices]
        m_j = Z_j.shape[0]
        c_j = d / (m_j * eps)
        logdet_j = logdet(I + c_j * Z_j.T @ Z_j)
        loss_comp += logdet_j * m_j / (2 * m)
    loss_expd, loss_comp = loss_expd.item(), loss_comp.item()
    print(loss_expd,loss_comp,loss_expd - loss_comp)
    return loss_comp-loss_expd, loss_expd, loss_comp



def covariance(X):
    return contract('ji...,jk...->ik...', X, X.conj())

def logdet(X):
    sgn, logdet = torch.linalg.slogdet(X)
    return sgn * logdet