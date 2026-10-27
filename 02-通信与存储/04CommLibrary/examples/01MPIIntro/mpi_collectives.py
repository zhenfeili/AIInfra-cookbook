import sys

import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if size != 4:
    if rank == 0:
        print("Run with exactly 4 processes.", file=sys.stderr, flush=True)
    sys.exit(1)  # 所有 Rank 都在开始通信前退出

factor = np.array([10 if rank == 0 else 0], dtype=np.intc)
comm.Bcast([factor, 1, MPI.INT], root=0)  # 每个 Rank 都得到 [10]

input_data = np.arange(8, dtype=np.intc) if rank == 0 else None
local = np.zeros(2, dtype=np.intc)
sendbuf = [input_data, 2, MPI.INT] if rank == 0 else None
comm.Scatter(sendbuf, [local, 2, MPI.INT], root=0)
# Root 向每个 Rank 发送 2 个整数，自己也得到一块

local *= factor[0]  # 仅修改当前 Rank 的两个元素

output = np.zeros(8, dtype=np.intc) if rank == 0 else None
recvbuf = [output, 2, MPI.INT] if rank == 0 else None
comm.Gather([local, 2, MPI.INT], recvbuf, root=0)
# Root 从每个 Rank 收 2 个整数，按 Rank 排列；总容量为 4×2

if rank == 0:
    print("gathered:", *output.tolist(), flush=True)
