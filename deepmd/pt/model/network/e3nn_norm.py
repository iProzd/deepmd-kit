import torch
import torch.nn as nn
import e3nn.o3 as o3
from e3nn.o3 import Irreps


def _idx_l(irreps: Irreps, l: int):
    for i, (_, ir) in enumerate(irreps):
        if ir.l == l:
            return i
    return None


class PowerSpectrumL2(nn.Module):
    """
    输入 irreps 例: "128x0e + 64x1e + 64x2e"
    输出: 逐通道二次不变量 (0e)，包括 (0,0)->0, (1,1)->0, (2,2)->0 （只对存在的阶）
    """
    def __init__(self, irreps_in: str):
        super().__init__()
        self.ir_in = Irreps(irreps_in).simplify()

        instrs = []
        outs = Irreps([])
        # 为存在的每个 l 建一个 (l,l)->0 的逐通道收缩
        for l in (0, 1, 2):
            i = _idx_l(self.ir_in, l)
            if i is None:
                continue
            mul = self.ir_in[i].mul
            outs += Irreps(f"{mul}x0e")
            instrs.append((i, i, len(outs)-1, "uuu", False))

        # 若完全没有高阶或 0e，也允许为空
        self.ir_out = outs
        self.tp = o3.TensorProduct(
            self.ir_in, self.ir_in, self.ir_out,
            instructions=instrs,
            irrep_normalization="norm", path_normalization="element",
            internal_weights=False, shared_weights=True
        )

    def forward(self, x):
        return self.tp(x, x).sqrt()
