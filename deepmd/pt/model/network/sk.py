# SPDX-License-Identifier: LGPL-3.0-or-later
import torch

# import warnings


class SinkhornKnopp:
    """
    PyTorch 版高效 Sinkhorn-Knopp 算法.

    输入非负方阵 P, 通过迭代归一化行列, 将其转换为双随机矩阵 (Doubly Stochastic Matrix).
    """

    def __init__(
        self, max_iter: int = 1000, epsilon: float = 1e-2, check_interval: int = 10
    ) -> None:
        """
        Parameters
        ----------
        max_iter : int
            最大迭代次数.
        epsilon : float
            停止条件的容差阈值.
        check_interval : int
            每隔多少次迭代检查一次收敛条件(为了提高效率,不必每次都检查).
        """
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.check_interval = check_interval

        # 记录运行状态
        self._iterations = 0
        # self._stopping_condition = None
        # self._D1 = None  # 行缩放系数 r
        # self._D2 = None  # 列缩放系数 c

    # @torch.no_grad()
    def fit(self, x: torch.Tensor) -> [torch.Tensor, torch.Tensor]:
        """
        运行 Sinkhorn 算法.

        Parameters
        ----------
        x : torch.Tensor or array-like
            形状为 (N, N) 的非负方阵.

        Returns
        -------
        x_ds : torch.Tensor
            计算得到的双随机矩阵.
        """
        device = x.device
        dtype = x.dtype
        # N = x.shape[-1]

        # 2. 初始化 r 和 c
        # 形状设计:(..., N, 1)
        # 这样设计是为了方便直接利用 torch.matmul 进行批量矩阵乘法
        batch_shape = x.shape[:-1]

        # r: (..., N, 1)
        r = torch.ones(*batch_shape, 1, device=device, dtype=dtype)
        c = torch.ones(*batch_shape, 1, device=device, dtype=dtype)

        # P_t: (..., N, N) 预转置最后两维,用于列更新
        # 显式 transpose(-2, -1) 是为了配合 matmul
        x_t = x.transpose(-2, -1)

        self._iterations = 0

        # 3. 批量迭代
        for i in range(self.max_iter):
            self._iterations += 1

            # --- 步骤 A: 归一化列 (Column Update) ---
            # 公式: c = 1 / (x^T @ r)
            # P_t: (..., N, N), r: (..., N, 1) -> matmul -> (..., N, 1)
            # 这里的 @ 自动处理前面的 batch 维度

            c = 1.0 / (x_t @ r + 1e-12)

            # --- 步骤 B: 收敛检查 (Lazy Check) ---
            # 检查行和是否接近 1 (利用 row_sums = r * (x @ c))
            if i % self.check_interval == 0:
                # x @ c: (..., N, 1)
                # row_sums: (..., N, 1)
                row_sums = r * (x @ c)

                # 计算误差:我们需要检查整个 Batch 中最大的那个误差
                # err shape: scalar
                err = torch.max(torch.abs(row_sums - 1.0))

                if err < self.epsilon:
                    break

            # --- 步骤 C: 归一化行 (Row Update) ---
            # 公式: r = 1 / (x @ c)
            r = 1.0 / (x @ c + 1e-12)

        # 4. 构建结果
        # 利用广播机制:
        # P_ds = diag(r) @ x @ diag(c)
        # r: (..., N, 1)
        # x: (..., N, N)
        # c.transpose: (..., 1, N)
        # 结果: (..., N, N)

        # c 需要转置为行向量 (..., 1, N) 以便广播到每一行
        # x_ds = r * x * c.transpose(-2, -1)

        return r, c


# # --- 使用示例 ---
# if __name__ == "__main__":
#     # 1. 准备数据
#
#     print("\n--- GPU Batch 性能测试 ---")
#     B_large = 128
#     N_large = 128
#     large_P = torch.rand(B_large, N_large, N_large, device="cuda")
#
#     sk_gpu = SinkhornKnopp(epsilon=1e-4)
#
#     import time
#
#     torch.cuda.synchronize()
#     start = time.time()
#     sk_gpu.fit(large_P)
#     torch.cuda.synchronize()
#     print(f"处理形状 {large_P.shape} 耗时: {time.time() - start:.4f}s")
