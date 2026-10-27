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

    float weights[2] = {1.0f, 1.0f};  // 两个进程具有相同的初始参数
    float grad[2] = {1.0f + 2.0f * rank, 3.0f + 2.0f * rank};
    float grad_sum[2] = {0.0f, 0.0f};
    MPI_Allreduce(grad, grad_sum, 2, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    // 所有进程提供 grad，所有进程收到 [4, 8]；没有 root 参数

    const float learning_rate = 0.1f;
    for (int i = 0; i < 2; ++i) {
        float grad_mean = grad_sum[i] / size;  // 等量样本时求进程平均
        weights[i] -= learning_rate * grad_mean;
    }
    printf("rank %d: sum=[%.1f, %.1f], weights=[%.1f, %.1f]\n",
           rank, grad_sum[0], grad_sum[1], weights[0], weights[1]);

    MPI_Finalize();
    return 0;
}
