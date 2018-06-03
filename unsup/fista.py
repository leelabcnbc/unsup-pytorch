"""reimplements
https://github.com/torch/optim/blob/63994c78b2eef4266e62e88e0ae444ee0c37074d/fista.lua
"""

from math import sqrt
from torch import FloatTensor
import torch


def fista_ls(f, g, pl, x0: FloatTensor, *, verbose=False, L=0.1):
    # define params

    # x init should be a Tensor or Tensor.cuda.
    # f, g, pl takes Tensor and returns Python float or Tensor (NO Variable).

    maxiter = 50
    maxline = 20
    errthres = 1e-4
    L_step = 1.5
    niter = 0

    xkm: FloatTensor = x0

    # xkm is x_{k-1} in the original paper.
    yk: FloatTensor = xkm.clone()
    tk = 1.0  # momentum

    fval = float('inf')

    while niter < maxiter:
        # get function value and gradient w.r.t. input
        # fy should be float,
        # dfy should be a vanilla Tensor of same type as x0
        fy, dfy = f(yk, True)

        # line search.
        nline = 0
        linesearchdone = False

        xk = None  # simply for lint purpose.
        while not linesearchdone:
            # this solves p_L(y) in the paper, or Eq. (2.6).
            # check Eq. (2.3) and the solution below, the original implementation of pl
            # in https://github.com/koraykv/unsup/blob/54c7a7e0843049436ae3dcd20d9d10716c2ba5cb/FistaL1.lua#L76
            # is correct (I mean consistent with the paper).
            xk = pl(yk - 1 / L * dfy, L)
            # this minimizer can be the next xk

            # now check whether the line search should stop.
            f_xk = f(xk, False)
            # compute Q
            ply_m_y = xk - yk
            q2 = torch.sum(ply_m_y * dfy)
            q3 = L / 2 * torch.sum(ply_m_y ** 2)
            q = fy + q2 + q3

            if verbose:
                print('nline={:d} L={:g} fply={:g} Q={:g} fy={:g}, Q2={:g}, Q3={:g}'.format(nline, L, f_xk,
                                                                                        q, fy, q2, q3))

            if f_xk <= q:
                # stop
                linesearchdone = True
            elif nline >= maxline:
                # looks that nline is additional number of line search.
                # there must be at least one evaluation of line search in this implementation.
                linesearchdone = True
                xk = xkm
            else:
                L *= L_step

            nline += 1

        # FISTA
        # t_{k+1}
        tkp = (1 + sqrt(1 + 4 * tk * tk)) / 2
        yk = xk + (tk - 1) / tkp * (xk - xkm)
        xkm = xk
        tk = tkp

        # check convergence.
        fk = f(yk, False)
        gk = g(yk)

        fval, fval_last = fk + gk, fval
        if verbose:
            print('iter={:d}, eold={:.8g}, enew={:.8g}'.format(niter, fval_last, fval))
        niter += 1
        # whether we are done
        if abs(fval - fval_last) <= errthres or niter >= maxiter:
            return yk, L

    raise RuntimeError('should not be here')
