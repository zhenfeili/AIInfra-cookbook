import sys

import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if size != 2:
    if rank == 0:
        print("Run with exactly 2 processes.", file=sys.stderr, flush=True)
    sys.exit(1)  # 所有 Rank 都在开始通信前退出

weights = np.array([1.0, 1.0], dtype=np.float32)
grad = np.array([1.0 + 2.0 * rank, 3.0 + 2.0 * rank], dtype=np.float32)
grad_sum = np.zeros(2, dtype=np.float32)
comm.Allreduce([grad, 2, MPI.FLOAT], [grad_sum, 2, MPI.FLOAT], op=MPI.SUM)
# 所有 Rank 都提供 grad，也都收到 [4, 8]；没有 root 参数

grad_mean = grad_sum / np.float32(size)  # 等量样本时求进程平均
learning_rate = np.float32(0.1)
weights -= learning_rate * grad_mean  # 每个 Rank 更新自己的参数副本
print(
    f"rank {rank}: sum=[{grad_sum[0]:.1f}, {grad_sum[1]:.1f}], "
    f"weights=[{weights[0]:.1f}, {weights[1]:.1f}]",
    flush=True,
)
