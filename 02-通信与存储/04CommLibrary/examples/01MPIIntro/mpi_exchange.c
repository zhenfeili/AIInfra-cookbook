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

    int peer = 1 - rank;  // Rank 0 的伙伴是 1，Rank 1 的伙伴是 0
    int outgoing[2] = {rank * 10, rank * 10 + 1};
    int incoming[2] = {-1, -1};
    MPI_Request requests[2];
    MPI_Irecv(incoming, 2, MPI_INT, peer, 20, MPI_COMM_WORLD, &requests[0]);
    MPI_Isend(outgoing, 2, MPI_INT, peer, 20, MPI_COMM_WORLD, &requests[1]);

    int local_result = rank + 100;  // 不依赖通信缓冲区的本地计算
    MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);  // 两个请求都完成后再读写缓冲区
    printf("rank %d received [%d, %d], local=%d\n",
           rank, incoming[0], incoming[1], local_result);

    MPI_Finalize();
    return 0;
}
