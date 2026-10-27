#include <mpi.h>
#include <stdio.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size != 4) {
        if (rank == 0) fprintf(stderr, "Run with exactly 4 processes.\n");
        MPI_Finalize();
        return 1;
    }

    int factor = (rank == 0) ? 10 : 0;
    MPI_Bcast(&factor, 1, MPI_INT, 0, MPI_COMM_WORLD);
    // 所有进程调用同一次广播；之后各自的 factor 都是 10

    int input[8] = {0};
    if (rank == 0) {
        for (int i = 0; i < 8; ++i) input[i] = i;
    }
    int local[2] = {0, 0};
    MPI_Scatter(input, 2, MPI_INT, local, 2, MPI_INT, 0, MPI_COMM_WORLD);
    // sendcount=2 表示发给每个进程 2 个元素，Root 也得到一块

    for (int i = 0; i < 2; ++i) local[i] *= factor;

    int output[8] = {0};
    MPI_Gather(local, 2, MPI_INT, output, 2, MPI_INT, 0, MPI_COMM_WORLD);
    // recvcount=2 表示从每个进程收 2 个元素，Root 总容量须为 4×2

    if (rank == 0) {
        printf("gathered:");
        for (int i = 0; i < 8; ++i) printf(" %d", output[i]);
        printf("\n");
    }
    MPI_Finalize();
    return 0;
}
