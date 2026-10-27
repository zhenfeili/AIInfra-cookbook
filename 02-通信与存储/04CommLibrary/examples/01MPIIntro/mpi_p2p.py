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

x = np.zeros(3, dtype=np.float32)  # 每个 Rank 独立持有形状 [3] 的数组
if rank == 0:
    x[:] = [1.0, 2.0, 3.0]
    comm.Send([x, 3, MPI.FLOAT], dest=1, tag=10)
    # 返回后可以修改发送缓冲区 x
else:
    comm.Recv([x, 3, MPI.FLOAT], source=0, tag=10)
    # 返回后接收缓冲区 x 已可读取
    print(f"rank 1 received: {x.tolist()}", flush=True)
