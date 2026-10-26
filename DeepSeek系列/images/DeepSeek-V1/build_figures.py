"""Rebuild the four authored SVG figures: python3 build_figures.py.

Requires numpy and matplotlib with a Chinese font. overview.svg is rendered
separately from overview.mmd using Mermaid 10.9.3. All illustrations are original;
source equations/data: https://arxiv.org/html/2401.02954v1 (see Markdown captions).
"""
from pathlib import Path
from html import escape
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

OUT = Path(__file__).resolve().parent
BLUE, GREEN, ORANGE = "#356da8", "#287c64", "#ba6831"
INK, MUTED, BORDER = "#183049", "#546778", "#c8d4df"
available = {f.name for f in font_manager.fontManager.ttflist}
font = next((x for x in ["Arial Unicode MS", "Noto Sans CJK SC", "Microsoft YaHei", "PingFang SC"] if x in available), "sans-serif")
plt.rcParams.update({"font.family": font, "font.size": 12, "svg.fonttype": "none",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.labelcolor": INK, "text.color": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.unicode_minus": False})


class SVG:
    def __init__(self, w, h, title):
        self.w, self.h = w, h
        self.items = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img">',
                      f'<title>{escape(title)}</title>',
                      '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8 Z" fill="#546778"/></marker></defs>',
                      f'<rect width="{w}" height="{h}" fill="white"/>']

    def text(self, x, y, value, size=20, color=INK, anchor="middle", weight=400):
        self.items.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" font-weight="{weight}" fill="{color}" font-family="Arial Unicode MS, Noto Sans CJK SC, PingFang SC, sans-serif">{escape(value)}</text>')

    def box(self, x, y, w, h, lines, fill="#eef4fb", stroke=BORDER, size=20):
        self.items.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" stroke="{stroke}"/>')
        step=size*1.45
        start=y+h/2-(len(lines)-1)*step/2+size*.35
        for i, line in enumerate(lines):
            self.text(x+w/2, start+i*step, line, size)

    def line(self, pts, arrow=True, color=MUTED):
        coords=" ".join(f'{x},{y}' for x,y in pts)
        marker=' marker-end="url(#arrow)"' if arrow else ''
        self.items.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="1.8"{marker}/>')

    def save(self, name):
        (OUT/name).write_text("\n".join(self.items+["</svg>"]), encoding="utf-8")


def compute_diagram():
    s=SVG(1000,660,"MHA 的张量形状与每 token 训练计算量；教学推导")
    s.text(500,36,"先算单层前向，再折算成每 token 训练 FLOPs",25,weight=600)
    s.text(500,67,"行向量约定 · 输入 X：[S,d] · m = 8d/3 · H×dₕ = d · 1 次乘加 = 2 FLOPs",18,MUTED)
    s.text(28,110,"① 参数相关的线性投影",21,anchor="start",weight=600)
    s.box(30,130,180,72,["输入 X","[S,d]"])
    s.box(265,130,360,72,["Q、K、V、输出投影","4 个 [d,d] 权重"])
    s.box(685,130,280,72,["4 × 2Sd²","= 8Sd² FLOPs"],fill="#eaf5f0")
    s.line([(210,166),(265,166)]);s.line([(625,166),(685,166)])
    s.box(30,225,180,72,["输入 X","[S,d]"])
    s.box(265,225,360,72,["SwiGLU：两条上投影 + 下投影","[S,d] → 两个 [S,m] → [S,d]"])
    s.box(685,225,280,72,["3 × 2Sdm","= 16Sd² FLOPs"],fill="#eaf5f0")
    s.line([(210,261),(265,261)]);s.line([(625,261),(685,261)])
    s.text(28,340,"② 不增加参数、但消耗算力的注意力交互",21,anchor="start",weight=600)
    s.box(30,365,270,85,["单头 QKᵀ","[S,dₕ] × [dₕ,S] → [S,S]"])
    s.box(365,365,300,85,["softmax 后乘 V","[S,S] × [S,dₕ] → [S,dₕ]"])
    s.box(725,365,240,85,["两次乘法，H 个头","4S²d FLOPs"],fill="#fff2e7")
    s.line([(300,407),(365,407)]);s.line([(665,407),(725,407)])
    s.box(30,490,935,100,["单层前向：24Sd² + 4S²d", "× 约 3（前向 + 反向）× L 层 ÷ S token  →  M = 72Ld² + 12LdS"],fill="#eaf5f0",size=22)
    s.text(500,625,"式 (2) 的 MHA 计数口径；省略词表、归一化等开销，不直接作为 67B GQA 的精确算量。",18,MUTED)
    s.save("compute-shapes.svg")


def gqa_diagram():
    s=SVG(1000,610,"DeepSeek LLM 67B 的 GQA 投影尺寸与组内共享")
    s.text(500,36,"67B：64 个 Q 头，8 组 K/V，每组服务 8 个 Q 头",25,weight=600)
    s.box(340,65,320,64,["输入 X：[S,8192]"],size=22)
    s.line([(420,129),(420,150),(245,150),(245,170)])
    s.line([(580,129),(580,150),(755,150),(755,170)])
    s.box(40,170,410,100,["WQ：[8192,8192]", "Q：[S,8192] → [S,64,128]"])
    s.box(550,170,410,100,["WK、WV：各 [8192,1024]", "K、V：各 [S,1024] → [S,8,128]"],fill="#eaf5f0")
    s.text(500,312,"展开第 0 组：每个头独立打分和输出，共享同一份 K₀、V₀",20)
    for i in range(8):
        x=25+i*122
        s.box(x,340,96,72,[f"头 {i}",f"Q{i} → O{i}"],size=18)
        s.line([(x+48,446),(x+48,412)])
    s.line([(73,446),(927,446)],False)
    s.line([(500,483),(500,446)],False)
    s.box(310,483,380,58,["共享缓存 K₀、V₀：各 [S,128]"],fill="#eaf5f0",size=20)
    s.text(500,576,"组 g 服务 Q 头 8g … 8g+7；组内箭头表示读取 K/V，K/V 无需物理复制 8 份。",18,MUTED)
    s.save("gqa-shapes.svg")


def allocation_plot():
    fig,axs=plt.subplots(1,2,figsize=(11,4.5),layout="constrained")
    x=np.geomspace(.2,5,200)
    # Teaching-only separable loss: coefficients place its minimum at x=1.
    loss=.5243*x**(-.4757)+.4757*x**.5243
    ax=axs[0]
    ax.plot(x,loss,color=BLUE,lw=2.5)
    ax.scatter([1],[1],color=GREEN,zorder=4,s=45)
    ax.axvline(1,color=BORDER,ls='--')
    ax.set_xscale('log',base=2);ax.set_xticks([.25,.5,1,2,4],['¼','½','1','2','4'])
    ax.set_ylim(.95,1.43)
    ax.set_xlabel('相对模型规模 M / M*')
    ax.set_ylabel('教学损失（任意单位，越低越好）')
    ax.set_title('固定 C：M 增大，D = C/M 减小',fontsize=14,pad=14)
    ax.annotate('配比最优点',xy=(1,1),xytext=(1.5,1.1),arrowprops={'arrowstyle':'->','color':MUTED},color=GREEN)
    ax.text(.04,.91,'左：模型偏小\n右：训练 token 偏少',transform=ax.transAxes,va='top',fontsize=11)
    ax.grid(axis='y',alpha=.18)
    ax=axs[1];r=np.linspace(1,10,200)
    ax.plot(r,r**.5243,label='模型规模 M：r^0.5243',color=BLUE,lw=2.5)
    ax.plot(r,r**.4757,label='数据规模 D：r^0.4757',color=GREEN,lw=2.5,ls='--')
    ax.scatter([10,10],[10**.5243,10**.4757],color=[BLUE,GREEN],s=25)
    ax.text(10.15,10**.5243,'3.34×',color=BLUE,fontsize=11,va='center')
    ax.text(10.15,10**.4757,'2.99×',color=GREEN,fontsize=11,va='center')
    ax.set_xlim(1,12);ax.set_xticks([1,2,4,6,8,10]);ax.set_ylim(.9,3.7)
    ax.set_xlabel('算力预算倍率 r = C / C₀');ax.set_ylabel('相对 C₀ 的最优规模倍率')
    ax.set_title('预算扩大：模型与数据一起增长',fontsize=14,pad=14)
    ax.legend(loc='upper left',frameon=False,fontsize=11);ax.grid(alpha=.18)
    fig.savefig(OUT/'compute-allocation.svg',metadata={'Date':None},bbox_inches='tight',pad_inches=.22)
    plt.close(fig)


def benchmark_plot():
    # Exact values from Table 5, arXiv:2401.02954v1. Do not substitute V2 reevaluations.
    names=['MMLU · 5-shot','GSM8K · 8-shot','HumanEval · 0-shot','HellaSwag · 0-shot']
    llama=[69.0,58.4,28.7,84.0]; deepseek=[71.3,63.4,42.7,84.0]
    fig,ax=plt.subplots(figsize=(9,4),layout='constrained')
    y=np.arange(4)
    ax.barh(y-.16,llama,.29,color='#9aaabd',label='LLaMA-2 70B Base')
    ax.barh(y+.16,deepseek,.29,color=BLUE,label='DeepSeek LLM 67B Base')
    for ys,values in [(y-.16,llama),(y+.16,deepseek)]:
        for yy,v in zip(ys,values):ax.text(v+1,yy,f'{v:.1f}',va='center',fontsize=11)
    ax.set_yticks(y,names);ax.invert_yaxis();ax.set_xlim(0,100)
    ax.set_xlabel('分数（%，越高越好；不同任务不能直接平均）')
    ax.legend(loc='lower left',bbox_to_anchor=(0,1.02),ncol=2,frameon=False,fontsize=10)
    ax.set_axisbelow(True);ax.grid(axis='x',alpha=.15)
    fig.savefig(OUT/'base-benchmarks.svg',metadata={'Date':None},bbox_inches='tight',pad_inches=.18)
    plt.close(fig)


if __name__ == '__main__':
    compute_diagram();gqa_diagram();allocation_plot();benchmark_plot()
    print('Generated compute-shapes.svg, gqa-shapes.svg, compute-allocation.svg, base-benchmarks.svg')
