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

peer = 1 - rank  # Rank 0 的伙伴是 1，Rank 1 的伙伴是 0
outgoing = np.array([rank * 10, rank * 10 + 1], dtype=np.intc)
incoming = np.full(2, -1, dtype=np.intc)
requests = [
    comm.Irecv([incoming, 2, MPI.INT], source=peer, tag=20),
    comm.Isend([outgoing, 2, MPI.INT], dest=peer, tag=20),
]

local_result = rank + 100  # 不依赖通信缓冲区的本地计算
MPI.Request.Waitall(requests)  # 两个请求都完成后再读写相关缓冲区
print(f"rank {rank} received {incoming.tolist()}, local={local_result}", flush=True)
