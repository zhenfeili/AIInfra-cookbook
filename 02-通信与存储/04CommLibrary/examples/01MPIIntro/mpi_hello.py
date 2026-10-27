from mpi4py import MPI  # 默认在首次导入 MPI 模块时初始化 MPI

comm = MPI.COMM_WORLD  # 本次启动的所有 MPI 进程
rank = comm.Get_rank()  # 当前进程在 comm 中的编号
size = comm.Get_size()  # comm 中的进程总数
host = MPI.Get_processor_name()  # 运行位置的名称

print(f"Hello: rank {rank}/{size} on {host}", flush=True)
# 默认在解释器正常退出时由 mpi4py 结束 MPI 使用
