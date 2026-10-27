"""Rebuild original MPI chapter diagrams as 300 dpi PNG and editable SVG.

Run: python3 01MPIIntro_book_figures.py
Requires matplotlib. Chinese font: Noto Sans CJK SC, Microsoft YaHei,
PingFang SC/HK, or Arial Unicode MS. Outputs go to the parent images directory.
Existing repository illustrations are referenced unchanged by the chapter.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

OUT = Path(__file__).resolve().parent.parent
AVAILABLE = {f.name for f in font_manager.fontManager.ttflist}
FONT = next((name for name in ['Noto Sans CJK SC', 'Microsoft YaHei',
                             'PingFang SC', 'PingFang HK', 'Arial Unicode MS']
             if name in AVAILABLE), None)
if FONT is None:
    raise RuntimeError('Install Noto Sans CJK SC or another Chinese font listed above.')
plt.rcParams.update({'font.family': FONT, 'font.size': 15,
                     'axes.unicode_minus': False, 'svg.fonttype': 'path'})
INK, MUTED, LINE = '#233449', '#526479', '#ced8e3'
COLORS = ['#268566', '#c45470', '#be7e25', '#356eb7']
PALE = ['#edf7f2', '#fceff3', '#fff5e7', '#eff5fc']


def canvas(title, height=6):
    fig, ax = plt.subplots(figsize=(10, height), dpi=150)
    fig.subplots_adjust(left=.02, right=.98, top=.98, bottom=.02)
    ax.set(xlim=(0, 100), ylim=(0, 100))
    ax.axis('off')
    ax.text(3, 97, title, fontsize=19, color=INK, va='top', weight='bold')
    return fig, ax


def text(ax, x, y, label, size=15, color=INK, **kwargs):
    return ax.text(x, y, label, fontsize=size, color=color, ha='center', va='center', **kwargs)


def box(ax, x, y, w, h, label='', fill='#f5f8fc', edge=LINE, size=15):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.2,rounding_size=1',
                              linewidth=1.2, facecolor=fill, edgecolor=edge))
    if label:
        text(ax, x+w/2, y+h/2, label, size)


def arrow(ax, start, end, label='', color=INK, dashed=False):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=14,
                                color=color, linewidth=1.4,
                                linestyle='--' if dashed else '-'))
    if label:
        text(ax, (start[0]+end[0])/2, (start[1]+end[1])/2+3, label, 13,
             bbox={'facecolor':'white','edgecolor':'none','pad':1})


def save(fig, stem):
    fig.savefig(OUT/f'{stem}.png', dpi=300, facecolor='white')
    fig.savefig(OUT/f'{stem}.svg', facecolor='white')
    plt.close(fig)


def architecture():
    fig, ax = canvas('通信语义、库实现与传输路径')
    box(ax, 5, 74, 90, 13, '应用 / 训练框架：任务划分、张量布局、计算与更新', '#eaf2fb')
    box(ax, 5, 39, 42, 22, 'MPI 标准接口\nOpen MPI / MPICH 实现', PALE[0])
    box(ax, 53, 39, 42, 22, '其他通信接口与后端\nNCCL / Gloo / HCCL 等', PALE[3])
    arrow(ax, (26, 74), (26, 62))
    arrow(ax, (74, 74), (74, 62))
    text(ax, 50, 67, '按应用与设备选择路径', 13)
    box(ax, 5, 8, 90, 18, '由实现与环境选择：同机共享内存 / 网络传输 / 设备互联\nCPU 内存、GPU 显存、网卡与互联硬件', '#f7f7f5')
    arrow(ax, (26, 39), (26, 27))
    arrow(ax, (74, 39), (74, 27))
    text(ax, 50, 3, '并列路径表示可选方案；不存在“所有通信都必须经过 MPI”的要求。', 12, MUTED)
    save(fig, '01MPIIntro_book_stack')


def ranks():
    fig, ax = canvas('物理位置与通信域内编号是不同维度', 6.7)
    for node, x in enumerate([3, 52]):
        box(ax, x, 56, 45, 30, fill='#f7f9fb')
        text(ax, x+22.5, 81, f'机器 {node}：独立地址空间中的两个进程', 14)
        for j in range(2):
            rank=node*2+j
            box(ax, x+3+j*21, 60, 18, 15,
                f'世界 Rank {rank}\n设备 {j}（示例绑定）', PALE[rank], COLORS[rank], 12)
    text(ax, 50, 49, 'MPI_COMM_WORLD = {0, 1, 2, 3}；按世界 Rank 奇偶划分', 15)
    for x, name, world in [(3,'A',[0,2]), (52,'B',[1,3])]:
        box(ax, x, 15, 45, 26, fill='#fafbfd')
        text(ax, x+22.5, 36, f'通信域 {name}：可跨越机器边界', 15)
        for j, rank in enumerate(world):
            box(ax, x+3+j*21, 19, 18, 12,
                f'世界 Rank {rank}\n→ 组内 Rank {j}', PALE[rank], COLORS[rank], 13)
    text(ax, 50, 7, '设备编号在机器内使用；组内 Rank 在各自通信域内使用。', 14, MUTED)
    save(fig, '01MPIIntro_book_ranks')


def contexts():
    fig, ax = canvas('同一进程组，不同上下文：消息不能跨通信域匹配', 6)
    for y, name, value, color, pale in [(56,'comm_A','[1, 2]',COLORS[0],PALE[0]),
                                      (22,'comm_B','[9, 8]',COLORS[3],PALE[3])]:
        box(ax, 3, y, 94, 24, fill=pale, edge=color)
        text(ax, 15, y+18, name, 16, color)
        box(ax, 7, y+3, 25, 11, f'Rank 0：{value}', '#ffffff', size=14)
        box(ax, 67, y+3, 25, 11, f'Rank 1：{value}', '#ffffff', size=14)
        arrow(ax, (33,y+8.5), (66,y+8.5), 'source=0，dest=1，tag=10', color)
    text(ax, 50, 51, '相同成员、相同 Rank、相同 Tag，仍然属于两套消息匹配空间', 13, MUTED)
    text(ax, 50, 11, '接收方仍需提供兼容的数据类型，以及容纳 2 个元素的缓冲区。', 14, MUTED)
    save(fig, '01MPIIntro_book_contexts')


def collectives():
    fig, ax = canvas('相同输入：保留各份数据，还是逐元素归约？', 6.8)
    inputs=[[1,2],[3,4],[5,6],[7,8]]
    summed=[sum(row[i] for row in inputs) for i in range(2)]
    text(ax, 50, 87, '四个 Rank 分别输入长度为 2 的数组；求和结果为 [16, 20]', 14, MUTED)
    for rank, values in enumerate(inputs):
        box(ax, 4+rank*24, 72, 21, 10, f'Rank {rank}：{values}', PALE[rank], COLORS[rank], 13)
    gathered='[1, 2 | 3, 4 | 5, 6 | 7, 8]'
    configurations=[(3,39,'Gather',f'仅 Root 得到：\n{gathered}\n逻辑形状 [4, 2]，共 8 元素',0),
                    (52,39,'Reduce SUM',f'仅 Root 得到：\n{summed}\n形状 [2]，共 2 元素',3),
                    (3,5,'Allgather',f'每个 Rank 都得到：\n{gathered}\n逻辑形状 [4, 2]，共 8 元素',0),
                    (52,5,'Allreduce SUM',f'每个 Rank 都得到：\n{summed}\n形状 [2]，共 2 元素',3)]
    for x,y,title,body,c in configurations:
        box(ax,x,y,45,29,fill=PALE[c])
        text(ax,x+22.5,y+24,title,17,COLORS[c])
        text(ax,x+22.5,y+11,body,14,linespacing=1.5)
    save(fig, '01MPIIntro_book_collectives')


def overlap():
    fig, ax = canvas('通信重叠取决于数据何时就绪，以及通信能否持续推进', 6)
    # Three compute segments of 4 ms; corresponding communication costs 3, 3, 2 ms.
    def x(t): return 17+t*3.8
    for t in range(0,21,2):
        ax.plot([x(t),x(t)],[13,82],color='#e7edf3',lw=.8,zorder=0)
        text(ax,x(t),9,str(t),12,MUTED)
    text(ax,96,9,'ms',12,MUTED)
    for label, y in [('无重叠',85),('部分重叠',49)]: text(ax,7,y,label,14)
    for y,label in [(72,'反向计算'),(59,'通信'),(36,'反向计算'),(23,'通信')]:
        text(ax,8,y,label,13)
    for y in [72,36]:
        for i in range(3):
            ax.add_patch(Rectangle((x(i*4),y-4),4*3.8,8,fc=PALE[3],ec=COLORS[3],lw=1.2))
            text(ax,x(i*4+2),y,f'桶 {i} 梯度',12)
    for y,starts in [(59,[12,15,18]),(23,[4,8,12])]:
        for i,(start,length) in enumerate(zip(starts,[3,3,2])):
            ax.add_patch(Rectangle((x(start),y-4),length*3.8,8,fc=PALE[0],ec=COLORS[0],lw=1.2))
            text(ax,x(start+length/2),y,f'C{i}',13)
    text(ax,70,85,'12 ms 计算 + 8 ms 通信 = 20 ms',14)
    text(ax,70,49,'隐藏 6 ms 通信，阶段耗时 14 ms',14)
    text(ax,50,2,'教学假设，非实测：三桶就绪时间为 4、8、12 ms；忽略资源竞争。',12,MUTED)
    save(fig,'01MPIIntro_book_overlap')


if __name__ == '__main__':
    for draw in [architecture, ranks, contexts, collectives, overlap]:
        draw()
    print(f'Wrote 5 PNG/SVG pairs to {OUT}')
