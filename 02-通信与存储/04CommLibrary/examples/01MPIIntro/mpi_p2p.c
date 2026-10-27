#include <mpi.h>
#include <stdio.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size != 2) {
        if (rank == 0) fprintf(stderr, "Run with exactly 2 processes.\n");
        MPI_Finalize();
        return 1;
    }

    float x[3] = {0.0f, 0.0f, 0.0f};  // 每个进程都有自己的 x
    if (rank == 0) {
        x[0] = 1.0f; x[1] = 2.0f; x[2] = 3.0f;
        MPI_Send(x, 3, MPI_FLOAT, 1, 10, MPI_COMM_WORLD);
        // 发给 Rank 1，Tag 为 10；返回后可以修改本进程的 x
    } else {
        MPI_Recv(x, 3, MPI_FLOAT, 0, 10, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        // 从 Rank 0 接收 Tag 为 10 的消息；返回后 x 已可读取
        printf("rank 1 received: [%.1f, %.1f, %.1f]\n", x[0], x[1], x[2]);
    }
    MPI_Finalize();
    return 0;
}
