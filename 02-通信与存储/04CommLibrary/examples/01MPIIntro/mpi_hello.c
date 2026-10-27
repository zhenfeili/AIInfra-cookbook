#include <mpi.h>    // MPI 的类型、常量和函数声明
#include <stdio.h>  // printf()

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);  // 每个进程初始化自己的 MPI 环境

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // 查询当前进程的 Rank
    MPI_Comm_size(MPI_COMM_WORLD, &size);  // 查询进程总数

    char host[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(host, &name_len);  // 查询运行位置的名称
    printf("Hello: rank %d/%d on %s\n", rank, size, host);

    MPI_Finalize();  // 本进程结束 MPI 使用；此后不再进行 MPI 通信
    return 0;
}
