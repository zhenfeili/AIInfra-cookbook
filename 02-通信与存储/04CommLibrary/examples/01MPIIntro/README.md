# MPI 章节配套示例

五个 Python 文件与[章节正文](../../01MPIIntro.md)中的完整程序一致，使用 `mpi4py + NumPy` 和 CPU 连续数组，不需要 GPU。同名 C 文件保留为接口对照，数值示例与 Python 版本一致。

| Python 正文源码 | C 对照源码 | 进程数 | 观察内容 |
| --- | --- | --- | --- |
| [mpi_hello.py](./mpi_hello.py) | [mpi_hello.c](./mpi_hello.c) | 4 | Rank、进程总数与主机名称 |
| [mpi_p2p.py](./mpi_p2p.py) | [mpi_p2p.c](./mpi_p2p.c) | 2 | Rank 0 向 Rank 1 传送三个浮点数 |
| [mpi_exchange.py](./mpi_exchange.py) | [mpi_exchange.c](./mpi_exchange.c) | 2 | 两个 Rank 非阻塞交换两个整数 |
| [mpi_collectives.py](./mpi_collectives.py) | [mpi_collectives.c](./mpi_collectives.c) | 4 | 广播乘数、分发数据、计算后收集 |
| [mpi_gradient.py](./mpi_gradient.py) | [mpi_gradient.c](./mpi_gradient.c) | 2 | 梯度求和、求平均与一次参数更新 |

## 环境与运行

Linux 或 macOS 需要可用的 MPI 运行时与启动器。先按正文安装 Open MPI，或加载集群提供的 MPI 模块，再在本目录创建 Python 环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy mpi4py
mpiexec --version
python -c "from mpi4py import MPI; print(MPI.Get_library_version())"
```

`mpi4py` 实际加载的 MPI 库应与 `mpiexec` 来自同一套安装。后续 `python` 指已激活环境中的解释器。安装细节见 [mpi4py 官方说明](https://mpi4py.readthedocs.io/en/stable/install.html)。

运行梯度示例：

```bash
mpiexec -n 2 python mpi_gradient.py
```

预期输出如下，行顺序可能不同：

```text
rank 0: sum=[4.0, 8.0], weights=[0.8, 0.6]
rank 1: sum=[4.0, 8.0], weights=[0.8, 0.6]
```

其他文件按表中进程数运行，例如 `mpiexec -n 4 python mpi_hello.py`。Hello 允许任意正进程数，表中 4 是正文演示值；其余四个示例在进程数错误时，由 Rank 0 打印提示，所有进程以非零状态退出。

正文使用首字母大写的缓冲区通信方法：浮点数组用 `np.float32` / `MPI.FLOAT`，整数数组用 `np.intc` / `MPI.INT`。Python 的 MPI 初始化与正常退出清理由 `mpi4py` 默认完成，非阻塞请求仍须显式等待完成。Python 集合示例只在 Root 分配完整输入、输出；C 示例各进程都分配小数组，非 Root 对应参数不参与传输，两者通信结果一致。

## 自动验证

检查 Python 示例：

```bash
python verify_examples.py
```

检查器使用当前 Python 解释器，经 `mpiexec` 在各 Rank 执行原始源码，并核对正文代码一致性、输出、数组形状、数据类型、连续布局和未舍入的数值结果。另验证四个固定进程数程序在错误启动时退出。输出按行比较，不要求 Rank 打印顺序固定；单次运行限时 30 秒。预期最后输出 `All 5 Python examples and 4 invalid-count cases passed.`。

同时验证保留的 C 示例：

```bash
python verify_examples.py --with-c
```

额外以 `-std=c11 -Wall -Wextra -Werror -pedantic` 编译五份 C 源码，运行并核对相同预期输出，检查四个错误进程数案例。编译产物位于临时目录，结束后自动清理；最后再输出 `All 5 C examples and 4 invalid-count cases passed.`。也可单独编译运行：

```bash
mpicc -std=c11 -Wall -Wextra mpi_gradient.c -o mpi_gradient
mpiexec -n 2 ./mpi_gradient
```

若本机 Open MPI 提示可用 slots 不足，可仅为教学检查启用超额订阅：

```bash
MPIEXEC_FLAGS=--oversubscribe python verify_examples.py --with-c
```

可用 `MPIEXEC` 指定启动器路径，用 `MPICC` 指定 C 编译器包装器路径；启动器、编译器与 Python 加载的库应使用同一套 MPI 安装。

检查环境：macOS / Open MPI 5.0.7 / Python 3.10.0 / NumPy 2.2.6 / mpi4py 4.1.2。测试用于验证本机通信语义，不代表跨机或 GPU 性能测试。NumPy 缓冲区能够使用底层 MPI 的高效路径，但不能由此推断整段 Python 程序与 C 程序性能相同。

## 插图重建

本章新增图的源文件为[绘图脚本](../../images/sources/01MPIIntro_book_figures.py)。在已安装 Matplotlib 的 Python 环境中运行：

```bash
python ../../images/sources/01MPIIntro_book_figures.py
```

脚本输出五组 300 dpi PNG 与同名 SVG 至 `images/`，PNG 均为 3000 像素宽。需要安装 Noto Sans CJK SC、Microsoft YaHei、PingFang SC/HK 或 Arial Unicode MS 中至少一种中文字体。图内数据是教学示例或明确标注的假设，不是性能实测。
