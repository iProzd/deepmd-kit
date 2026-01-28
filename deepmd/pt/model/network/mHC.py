# SPDX-License-Identifier: LGPL-3.0-or-later
import math
from typing import (
    Optional,
    Union,
)

import torch
import torch.nn as nn

from deepmd.pt.model.network.layernorm import (
    RMSNorm,
)
from deepmd.pt.model.network.mlp import (
    MLPLayer,
)
from deepmd.pt.model.network.sk import (
    SinkhornKnopp,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.env import (
    DEFAULT_PRECISION,
    PRECISION_DICT,
)

device = env.DEVICE


class MHCCoefficients(nn.Module):
    """
    Compute token-wise mHC coefficients from x_stream.

    x_stream: (B,S,n,C)
    H_pre:    (B,S,n)
    H_post:   (B,S,k,n) k is the number of residuals
    H_res:    (B,S,n,n)
    """

    def __init__(
        self,
        dim: int,
        n_streams: int = 4,
        tmax: int = 20,  # paper uses 20
        alpha_init: float = 0.01,
        rms_eps: float = 1e-6,
        b_res_offdiag: float = -4.0,  # init H_res close to I
        num_res: int = 1,  # number of residuals, k
        post_residual: float = 0.1,
        precision: str = DEFAULT_PRECISION,
        trainable: bool = True,
        seed: Optional[Union[int, list[int]]] = None,
    ) -> None:
        super().__init__()
        if n_streams < 1:
            raise ValueError("n_streams must be >= 1")
        self.dim = int(dim)
        self.n = int(n_streams)
        self.k = num_res
        self.nc = self.n * self.dim
        self.tmax = int(tmax)
        self.precision = precision
        self.prec = PRECISION_DICT[self.precision]

        self.rms = RMSNorm(
            self.nc, eps=rms_eps, precision=precision, trainable=trainable
        )

        out_dim = (1 + self.k) * self.n + self.n * self.n
        self.phi = MLPLayer(
            self.nc,
            out_dim,
            precision=precision,
            seed=seed,
            trainable=trainable,
            bias=False,
        )
        # self.phi = nn.Linear(self.nc, out_dim, bias=False)

        self.alpha_pre = nn.Parameter(
            data=torch.tensor(float(alpha_init), dtype=self.prec, device=device)
        )
        self.alpha_post = nn.Parameter(
            data=torch.tensor(float(alpha_init), dtype=self.prec, device=device)
        )
        self.alpha_res = nn.Parameter(
            data=torch.tensor(float(alpha_init), dtype=self.prec, device=device)
        )
        self.post_residual = post_residual

        # init H_pre ≈ 1/n, H_post ≈ 1, H_res ≈ I
        if self.n == 1:
            b_pre_init = 10.0  # sigmoid(10)≈1
        else:
            p = 1.0 / self.n
            b_pre_init = math.log(p / (1.0 - p))  # logit(1/n)

        self.b_pre = nn.Parameter(
            data=torch.full(
                (self.n,), float(b_pre_init), dtype=self.prec, device=device
            )
        )
        self.b_post = nn.Parameter(
            data=torch.zeros((self.k, self.n), dtype=self.prec, device=device)
        )

        b_res = torch.full(
            (self.n, self.n), float(b_res_offdiag), dtype=self.prec, device=device
        )
        b_res.fill_diagonal_(0.0)
        self.b_res = nn.Parameter(data=b_res)

    def forward(
        self, x_stream: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_shape = x_stream.shape
        assert x_shape[-1] == self.n * self.dim

        # Eq.(7): flatten vec(x_l) then RMSNorm
        x_flat = x_stream
        x_norm = self.rms(x_flat)
        # x_norm = x_flat

        proj = self.phi(x_norm)  # (B*S, (1+k)n+n^2)
        dyn_pre, dyn_post, dyn_res = torch.split(
            proj, [self.n, self.k * self.n, self.n * self.n], dim=-1
        )

        # Eq.(7): add gating and bias
        Ht_pre = self.alpha_pre * dyn_pre + self.b_pre
        Ht_post = self.alpha_post * dyn_post.view(-1, self.k, self.n) + self.b_post
        Ht_res = self.alpha_res * dyn_res.view(-1, self.n, self.n) + self.b_res

        # Eq.(8): constraints
        H_pre = torch.sigmoid(Ht_pre).view([*x_shape[:-1], self.n])  # (B,S,n)
        H_post = (
            torch.sigmoid(Ht_post) * (2.0 / float(self.k)) * self.post_residual
        ).view([*x_shape[:-1], self.k, self.n])
        H_res = sinkhorn_knopp(Ht_res, n_iters=self.tmax).view(
            [*x_shape[:-1], self.n, self.n]
        )

        # H_pre = torch.ones([*x_shape[:-1], self.n], dtype=self.prec, device=device) / float(self.n)
        # H_post = torch.ones([*x_shape[:-1], self.k, self.n], dtype=self.prec, device=device) / float(self.k)
        # H_res = torch.eye(self.n, dtype=self.prec, device=device).view([1]*len(x_shape[:-1]) + [self.n, self.n]).expand([*x_shape[:-1], self.n, self.n])
        # from IPython import embed
        # embed()
        # print((H_res>0.9).sum() / (H_res.numel()/4))
        return H_pre, H_post, H_res


def sinkhorn_knopp(x: torch.Tensor, n_iters: int = 20) -> torch.Tensor:
    """
    Fast sinkhorn_knopp.
    Implements Eq.(9): alternate row/col normalization after exp().
    """
    x_exp = torch.exp(x)
    # from IPython import embed
    # embed()
    # x_exp = x - x.min(-1, keepdim=True)[0]
    sk = SinkhornKnopp(max_iter=n_iters, check_interval=2)
    r, c = sk.fit(x_exp)
    xd = r * x_exp * c.transpose(-2, -1)
    # print(sk._iterations)
    # if sk._iterations >=19:
    #     from IPython import embed
    #     embed()
    return xd


def sinkhorn_knopp_slow(logits: torch.Tensor, n_iters: int = 20) -> torch.Tensor:
    """
    Differentiable Sinkhorn-Knopp projection in log-space.
    Implements Eq.(9): alternate row/col normalization after exp().
    """
    log_p = logits - logits.amax(dim=(-2, -1), keepdim=True)  # stability

    for _ in range(int(n_iters)):
        log_p = log_p - torch.logsumexp(log_p, dim=-1, keepdim=True)  # rows -> sum 1
        log_p = log_p - torch.logsumexp(log_p, dim=-2, keepdim=True)  # cols -> sum 1

    return torch.exp(log_p)
