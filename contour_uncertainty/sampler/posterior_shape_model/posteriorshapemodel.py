"""
Posterior Shape Models : https://edoc.unibas.ch/30789/1/20140113141209_52d3e629d3417.pdf
"""

from typing import Optional, List
import torch


def pca(X: torch.Tensor, mu: Optional[torch.Tensor] = None, return_all: bool = False):
    """ Compute pca model for data.

    Args:
        X: Input data (N, P)
        mu: Optional mean to compute PCA (P, 1)

    Returns:
        mu (P, 1) and Q (P,P) arrays
    """
    X = X[..., None]  # (N, P, 1)

    mu = mu if mu is not None else X.mean(axis=0)  # (P, 1)

    # Compute covariance matrix
    diff = X.squeeze().T - mu
    cov = torch.einsum('ij,kj', diff, diff) / X.shape[0]

    # Compute and sort eigenvalues and eigenvectors
    eigenvalues, eigenvectors = torch.linalg.eig(cov)
    eigenvalues = eigenvalues.real
    eigenvectors = eigenvectors.real

    eigenvalues = torch.abs(eigenvalues)

    idx = eigenvalues.argsort().flip(0)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    D = torch.diag(torch.sqrt(eigenvalues))
    U = eigenvectors

    Q = torch.mm(U, D)

    if return_all:
        return mu, Q, cov, D, U
    else:
        return mu, Q


def posterior_shape_model(s_g: torch.Tensor, g_indices: List, mu: torch.Tensor, Q: torch.Tensor, sigma2: float = 1):
    """Computes the posterior shape model conditional distribution.

    Args:
        s_g: Array of partial input (to be sliced by g_indices) (P, 1)
        g_indices: Indices of partial data.
        mu: pca mean (P, 1)
        Q: pca Q (P, P)
        sigma2: Noise (slack) parameter

    Returns:
        mean (P, 1) and covariance matrix (P, P) of the output distribution.
    """
    eye = torch.eye(len(mu), device=s_g.device)

    # mu_g = mu[g_indices]  # (q, 1)
    # Q_g = Q[g_indices]  # (q, p)
    # s_g = s_g[g_indices]  # (q, 1)

    mu_mask = torch.zeros(len(mu), 1, device=s_g.device)
    mu_mask[g_indices] = 1

    q_mask = torch.zeros(len(mu), len(mu), device=s_g.device)
    q_mask[g_indices] = 1

    mu_g = mu * mu_mask
    Q_g = Q * q_mask
    s_g = s_g * mu_mask

    mu_c = mu + Q @ torch.inverse(Q_g.T @ Q_g + sigma2 * eye) @ Q_g.T @ (s_g - mu_g)  # (p, 1)
    cov_c = sigma2 * Q @ torch.inverse(Q_g.T @ Q_g + sigma2 * eye) @ Q.T  # (p,p)

    return mu_c, cov_c

