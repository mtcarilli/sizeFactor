# module for glm full model fit 

import numpy as np
import pandas as pd
import scipy

import scipy.stats as stats
from scipy.linalg import solve
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.sparse.linalg import cg
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from collections.abc import Iterable


def getDesignExtended(C : int,
                      R : np.array,
                      G : int,
                      filt : np.array =  None):
    '''
    Inputs
    -----
    C : conditions, number of unique conditions
    R : list or numpy array or integer with number of replicates per condition. If R is an integer, each condition is assumed to have R replicates.
    G : number of genes 
    filt : either None (geometric mean is taken over all genes), 
           or a one dimensional boolean array of length G (geometric taken over same genes for all samples),
           or two dimensional boolean array of size S x G (geometric mean taken over different genes for different samples)

    Outputs
    -----
    design : design matrix for extended fit over all genes 
    '''
    if filt is None:
        filt = np.full(G,True)

    if not isinstance(R, Iterable):
        R = [R]*C

    R = np.array(R)
    S = R.sum()

    # create sample design matrix 
    sample_design = np.zeros((S*(G+1), S),dtype=np.int8)

    for i in np.arange(S):
        sample_design[i*(G+1):(i+1)*(G+1),i] = 1


    # create condition/gene design matrix
    cond_gene_design = np.zeros((S*G + S,C*G))

    # add the geoMetric mean for all genes... 
    for s in np.arange(S):
        if filt.ndim == 1:
            filt_ = filt
        else:
            filt_ = filt[s]
        G_sum = sum(filt_)
        mat_to_add = np.identity(G)
        geoMean_row = np.zeros(G)
        geoMean_row[filt_] = np.ones(G_sum)/G_sum
        mat_to_add = np.append(mat_to_add, [geoMean_row],axis=0)
        cond_gene_design[(s)*(G+1):(s+1)*(G+1),0:G] = mat_to_add

    index = R[0]
    for c in np.arange(1,C):
        R_ = R[c]
        for r in np.arange(R_):
            print(r)
            mat_to_add = np.identity(G)
            mat_to_add = np.append(mat_to_add,[ np.ones(G)/G ],axis=0)
            cond_gene_design[(index)*(G+1):(index+1)*(G+1),c*(G):(c+1)*(G)] = mat_to_add
            index+=1

    # final design matrix
    design = np.concatenate((sample_design,cond_gene_design),axis=1)

    return(design)


def nb_nll(
    counts: np.ndarray, 
    mu: np.ndarray, 
    alpha: np.ndarray,
) -> float:
    r"""Neg log-likelihood of a negative binomial of parameters ``mu`` and ``alpha``.

    Mathematically, if ``counts`` is a vector of counting entries :math:`y_i`
    then the likelihood of each entry :math:`y_i` to be drawn from a negative
    binomial :math:`NB(\mu, \alpha)` is [1]

    .. math::
        p(y_i | \mu, \alpha) = \frac{\Gamma(y_i + \alpha^{-1})}{
            \Gamma(y_i + 1)\Gamma(\alpha^{-1})
        }
        \left(\frac{1}{1 + \alpha \mu} \right)^{1/\alpha}
        \left(\frac{\mu}{\alpha^{-1} + \mu} \right)^{y_i}.

    We return the negative log likelihood summed over all counts, where each count can have a different mean and dispersion parameter. 
    """
    alpha_inverse = 1 / alpha
    logbinom = gammaln(counts + alpha_inverse) - gammaln(counts + 1) - gammaln(alpha_inverse)
    term1 = - alpha_inverse * np.log(1 + alpha * mu) + counts * np.log(mu) - counts * np.log(alpha_inverse + mu)
    ll = logbinom + term1
    neg_ll = -np.sum(ll) 


    return neg_ll


def irls_solver(
    counts: np.ndarray,
    design_matrix: np.ndarray,
    disp: np.ndarray,
    min_mu: float = 0.5,
    beta_tol: float = 1e-10,
    min_beta: float = -30,
    max_beta: float = 30,
    optimizer: str = "L-BFGS-B",
    maxiter: int = 250,
    ridge_factor: float = 1e-6
) -> tuple:
    r"""
    Taken from: https://github.com/owkin/PyDESeq2/blob/d1d3c5c8950de4f4cb7f8d3553f67c509ee36b1a/pydeseq2/utils.py#L363 


    Fit a NB GLM wit log-link to predict counts from the design matrix.

    See equations (1-2) in the DESeq2 paper.

    Parameters
    ----------
    counts : ndarray
        Raw counts for a given gene.

    design_matrix : ndarray
        Design matrix.

    disp : float
        Gene-wise dispersion prior.

    min_mu : float
        Lower bound on estimated means, to ensure numerical stability.
        (default: ``0.5``).

    beta_tol : float
        Stopping criterion for IRWLS:
        :math:`\vert dev - dev_{old}\vert / \vert dev + 0.1 \vert < \beta_{tol}`.
        (default: ``1e-8``).

    min_beta : float
        Lower-bound on LFC. (default: ``-30``).

    max_beta : float
        Upper-bound on LFC. (default: ``-30``).

    optimizer : str
        Optimizing method to use in case IRLS starts diverging.
        Accepted values: 'BFGS' or 'L-BFGS-B'.
        NB: only 'L-BFGS-B' ensures that LFCS will
        lay in the [min_beta, max_beta] range. (default: ``'L-BFGS-B'``).

    maxiter : int
        Maximum number of IRLS iterations to perform before switching to L-BFGS-B.
        (default: ``250``).

    ridge_factor : float
        Shrinkage factor to be applied to weights. 
        (default: ``1e-6``).

    Returns
    -------
    beta: ndarray
        Fitted (basemean, lfc) coefficients of negative binomial GLM.

    mu: ndarray
        Means estimated from size factors and beta: :math:`\mu = s_{ij} \exp(\beta^t X)`.

    H: ndarray
        Diagonal of the :math:`W^{1/2} X (X^t W X)^-1 X^t W^{1/2}` covariance matrix.

    converged: bool
        Whether IRLS or the optimizer converged. If not and if dimension allows it,
        perform grid search.
    """
    assert optimizer in ["BFGS", "L-BFGS-B"]

    num_vars = design_matrix.shape[1]
    X = sp.csr_matrix(design_matrix)

    # if full rank, estimate initial betas for IRLS below
    y = np.log(counts + 0.1)
    beta_init, _  = spla.cg(X.T @ X, X.T @ y, rtol=1e-6, maxiter=1000)
    beta = beta_init

    dev = 1000.0
    dev_ratio = 1.0

    ridge_factor = np.diag(np.repeat(ridge_factor, num_vars)) # sp.diags([1e-6] * num_vars, offsets=0, format='csr')
    mu = np.maximum(np.exp(X @ beta), min_mu)

    converged = True
    i = 0

    while dev_ratio > beta_tol:
        W = mu / (1.0 + mu * disp)
        z = np.log(mu) + (counts - mu) / mu
        H = (X.T.multiply(W)) @ X + ridge_factor
        beta_hat = solve(H, X.T @ (W * z), assume_a="pos")
        i += 1

        if sum(np.abs(beta_hat) > max_beta) > 0 or i >= maxiter:
            # If IRLS starts diverging, use L-BFGS-B
            def f(beta: np.ndarray) -> float:
                # closure to minimize
                mu_ = np.maximum(np.exp(X @ beta), min_mu)
                return nb_nll(counts, mu_, disp) + 0.5 * (ridge_factor @ beta**2).sum()

            def df(beta: np.ndarray) -> np.ndarray:
                mu_ = np.maximum(np.exp(X @ beta), min_mu)
                return (
                    -X.T @ counts
                    + ((1 / disp + counts) * mu_ / (1 / disp + mu_)) @ X
                    + ridge_factor @ beta
                )

            res = minimize(
                f,
                beta_init,
                jac=df,
                method=optimizer,
                bounds=(
                    [(min_beta, max_beta)] * num_vars
                    if optimizer == "L-BFGS-B"
                    else None
                ),
            )

            beta = res.x
            mu = np.maximum( np.exp(X @ beta), min_mu)
            converged = res.success

        beta = beta_hat
        mu = np.maximum( np.exp(X @ beta), min_mu)
        # Compute deviation
        old_dev = dev
        # Replaced deviation with -2 * nll, as in the R code
        dev = -2 * nb_nll(counts, mu, disp)
        dev_ratio = np.abs(dev - old_dev) / (np.abs(dev) + 0.1)

    # Compute H diagonal (useful for Cook distance outlier filtering)
    # Calculate only the diagonal for X(XTWX)-1XT using einsum
    # This is numerically equivalent to the more expensive calculation
    # np.diag(X @ (X^T @ np.inv(X^T @ np.diag(W) @ X + lambda) @ X^T)
    # W = mu / (1.0 + mu * disp)

    # WX = X.multiply(W[:, None])
    # XTWX = WX.T @ X
    # A = XTWX + ridge_factor
    # A = sp.csr_matrix(A)
    # A_inv_XT = spla.spsolve(A, X.T)
    # H = np.einsum("ij,ji->i", X, A_inv_XT)

    # H = np.einsum(
    #     "ij,jk,ki->i", X, np.linalg.inv((X.T * W[None, :]) @ X + ridge_factor), X.T
    # )


    # W_sq = np.sqrt(W)
    # H = W_sq * H * W_sq

    # Return an UNthresholded mu (as in the R code)
    # Previous quantities are estimated with a threshold though
    mu = np.exp(X @ beta)
    return beta, mu, converged
