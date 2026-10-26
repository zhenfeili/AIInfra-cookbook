"""Build original V3 teaching diagrams and the paper-data ablation chart.

Run: python3 build_figures.py (requires matplotlib and a Chinese font).
Render overview.mmd separately with Mermaid 10.9.3 to overview.svg.
Source: https://arxiv.org/html/2412.19437v2 ; captions in DeepSeek-V3.md.
"""
from html import escape
from pathlib import Path

OUT = Path(__file__).resolve().parent
INK, MUTED = '#183049', '#546778'
BLUE, GREEN, ORANGE = '#356da8', '#287c64', '#ba6831'


class SVG:
    def __init__(self, w, h, title):
        self.items = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img">',
                      f'<title>{escape(title)}</title>',
                      '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8 Z" fill="#546778"/></marker></defs>',
                      f'<rect width="{w}" height="{h}" fill="white"/>']

    def text(self, x, y, text, size=20, color=INK, anchor='middle', weight=400):
        self.items.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" fill="{color}" font-weight="{weight}" font-family="Arial Unicode MS, Noto Sans CJK SC, PingFang SC, sans-serif">{escape(text)}</text>')

    def rect(self, x, y, w, h, fill='#eef4fb', stroke='#c8d4df', radius=7):
        self.items.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}"/>')

    def box(self, x, y, w, h, lines, fill='#eef4fb', size=20):
        self.rect(x,y,w,h,fill)
        start=y+h/2-(len(lines)-1)*size*1.45/2+size*.35
        for i,line in enumerate(lines):
            self.text(x+w/2,start+i*size*1.45,line,size)

    def line(self, pts, arrow=True, dashed=False):
        end=' marker-end="url(#arrow)"' if arrow else ''
        dash=' stroke-dasharray="6 5"' if dashed else ''
        points=' '.join(f'{x},{y}' for x,y in pts)
        self.items.append(f'<polyline points="{points}" fill="none" stroke="{MUTED}" stroke-width="2"{end}{dash}/>')

    def save(self,name):
        (OUT/name).write_text('\n'.join(self.items+['</svg>']),encoding='utf-8')


def routing():
    s=SVG(1000,765,'无辅助损失均衡：偏置影响选择，原始分数决定权重')
    s.text(500,38,'把“选谁”与“输出占比”分开',26,weight=600)
    s.box(300,65,400,62,['亲和度 s = sigmoid（E uₜ）'])
    s.line([(390,127),(390,150),(250,150),(250,180)])
    s.line([(610,127),(610,150),(750,150),(750,180)])
    s.box(35,180,430,100,['选择分支：s + b → Top-K','得到专家集合 Sₜ'],fill='#fff2e7')
    s.box(535,180,430,100,['选中专家的权重：只读取原始 s','gᵢₜ = sᵢₜ / Σⱼ∈Sₜ sⱼₜ'],fill='#eaf5f0')
    s.line([(465,230),(535,230)],dashed=True)
    s.text(500,310,'虚线传递选中集合；偏置 b 不加进混合权重',19,MUTED)
    s.box(35,335,430,90,['按整个训练 batch 统计专家负载','过载：bᵢ − γ；欠载：bᵢ + γ'],fill='#fff2e7')
    s.line([(250,280),(250,335)])
    s.line([(35,380),(15,380),(15,155),(75,155),(75,180)],dashed=True)
    s.text(730,375,'下一训练步使用更新后的偏置',19,ORANGE)
    s.text(500,466,'教学示例：4 个专家选 2 个（实际 V3 为 256 选 8）',21,weight=600)
    labels=['专家','A','B','C','D']
    rows=[['亲和度 s','0.80','0.70','0.40','0.30'],
          ['偏置 b','−0.40','−0.30','+0.20','+0.25'],
          ['选择分数 s+b','0.40','0.40','0.60 ✓','0.55 ✓'],
          ['混合权重 g','0','0','4/7','3/7']]
    widths=[230,170,170,170,170]
    for r,row in enumerate([labels]+rows):
        x=45
        for c,(v,w) in enumerate(zip(row,widths)):
            fill='#eaf5f0' if c>=3 and r>=3 else ('#eef4fb' if r==0 else '#f8fafc')
            s.rect(x,490+r*42,w,42,fill,radius=0)
            s.text(x+w/2,517+r*42,v,19)
            x+=w
    s.text(500,735,'不加偏置时选 A、B；加偏置后选 C、D，权重仍由 0.40 与 0.30 归一化得到。',18,MUTED)
    s.save('loss-free-routing.svg')


def mtp():
    s=SVG(1000,800,'MTP 的因果关系与张量形状；V3 实际 D=1')
    s.text(500,38,'MTP 训练：多预测一步，但不读取该步的目标 token',25,weight=600)
    s.text(500,72,'列向量 · 实际 D = 1 · 图中只展开位置 i；Transformer 沿序列做因果注意力',18,MUTED)
    s.box(35,105,420,72,['主模型读取 t₁ … tᵢ','得到 hᵢ⁰：[7168,1]'])
    s.box(545,105,420,72,['训练语料提供真实 tᵢ₊₁','共享 Embedding：[7168,1]'],fill='#eaf5f0')
    s.line([(245,177),(245,215)])
    s.line([(755,177),(755,215)])
    s.box(35,215,420,58,['RMSNorm（hᵢ⁰）'])
    s.box(545,215,420,58,['RMSNorm（Emb（tᵢ₊₁））'],fill='#eaf5f0')
    s.line([(245,273),(245,307),(430,307),(430,340)])
    s.line([(755,273),(755,307),(570,307),(570,340)])
    s.box(225,340,550,90,['拼接：[14336,1]','M₁：[7168,14336] → h′ᵢ¹：[7168,1]'])
    s.line([(500,430),(500,465)])
    s.box(225,465,550,80,['因果 Transformer 块 TRM₁','hᵢ¹：[7168,1]'])
    s.line([(500,545),(500,580)])
    s.box(225,580,550,75,['共享输出头 → 词表概率 P¹ᵢ₊₂','交叉熵目标：真实 tᵢ₊₂'],fill='#eaf5f0')
    s.box(35,685,930,75,['主任务：hᵢ⁰ → 预测 tᵢ₊₁；MTP：hᵢ⁰ + 已知 tᵢ₊₁ → 预测 tᵢ₊₂。','训练时用真实前序 token；推测解码时换成已生成的候选，再由主模型验证。'],fill='#fff2e7',size=19)
    s.save('mtp-causal.svg')


def dualpipe():
    s=SVG(1000,640,'DualPipe 的重叠直觉；非真实调度表或耗时测量')
    s.text(500,38,'DualPipe：把一块的通信放进另一块的计算窗口',25,weight=600)
    s.text(500,74,'前向 F 与反向 B 来自不同 micro-batch；仍须遵守各自的数据依赖',18,MUTED)
    s.text(35,120,'层内依赖',21,anchor='start',weight=600)
    labels=['Attention','Dispatch','专家 MLP','Combine']
    fills=['#eef4fb','#fff2e7','#eef4fb','#fff2e7']
    for j,(label,fill) in enumerate(zip(labels,fills)):
        s.box(35+j*245,145,195,64,[label],fill)
        if j<3:s.line([(230+j*245,177),(280+j*245,177)])
    s.text(500,244,'反向计算进一步分为：输入梯度 Bₓ（供前层继续反传）与权重梯度 B𝓌',19,MUTED)
    s.text(35,292,'两个可重叠的工作窗口（仅示意）',21,anchor='start',weight=600)
    s.text(115,352,'计算资源',19)
    s.text(115,423,'通信资源',19)
    s.box(220,315,320,60,['反向块 B：已就绪的计算'],fill='#eaf5f0',size=19)
    s.box(555,315,410,60,['前向块 F：专家 MLP 等计算'],size=19)
    s.box(220,390,320,60,['前向块 F：Dispatch'],fill='#fff2e7',size=19)
    s.box(555,390,410,60,['反向块 B：已就绪的通信'],fill='#fff2e7',size=19)
    s.line([(220,475),(965,475)])
    s.text(945,505,'时间 →',18,MUTED)
    s.box(35,535,930,65,['全局再用双向流水线：从两端注入 micro-batch，减少填充与排空气泡。'],fill='#f5f7fa',size=20)
    s.text(500,625,'框宽不代表时间，图中不是论文图 4/5 的调度复刻；完整调度还包含 PP 通信和同步点。',17,MUTED)
    s.save('dualpipe-overlap.svg')


def fp8():
    s=SVG(1000,805,'FP8 的分组缩放与高精度累加；矩阵形状为教学尺寸')
    s.text(500,38,'FP8：小组各用自己的缩放，长点积分段高精度累加',25,weight=600)
    s.text(500,72,'图用 Y = XWᵀ 展示前向 GEMM：X [4,256]，W [256,256] → Y [4,256]',18,MUTED)
    s.text(245,118,'激活 X：每 token、每 128 通道一组',20,BLUE)
    for r in range(4):
        for c in range(2):
            s.rect(35+c*210,145+r*39,210,39,'#eaf5f0' if (r+c)%2 else '#eef4fb',radius=0)
            s.text(140+c*210,171+r*39,f'缩放 a{r+1},{c+1}',18)
    s.text(245,333,'每个小格：1 × 128；离群值只影响本组',18,MUTED)
    s.text(750,118,'权重 W：每 128 × 128 块一组',20,BLUE)
    for r in range(2):
        for c in range(2):
            s.rect(540+c*210,145+r*78,210,78,'#fff2e7' if (r+c)%2 else '#eef4fb',radius=0)
            s.text(645+c*210,190+r*78,f'缩放 w{r+1},{c+1}',20)
    s.text(750,333,'每个大格：128 × 128；图按分组示意',18,MUTED)
    s.text(35,385,'一个输出 yₘₙ 的点积，沿输入通道 K 分成两段',21,anchor='start',weight=600)
    s.box(35,410,430,85,['第 1 组：128 个元素的乘加','Tensor Cores：FP8 输入、部分累加'])
    s.box(535,410,430,85,['第 2 组：128 个元素的乘加','Tensor Cores：FP8 输入、部分累加'])
    s.line([(250,495),(250,530)])
    s.line([(750,495),(750,530)])
    s.box(35,530,430,65,['转到 CUDA Cores，乘 a(m,1) × w(b,1)'],fill='#eaf5f0',size=18)
    s.box(535,530,430,65,['转到 CUDA Cores，乘 a(m,2) × w(b,2)'],fill='#eaf5f0',size=18)
    s.line([(250,595),(250,620),(430,620),(430,645)])
    s.line([(750,595),(750,620),(570,620),(570,645)])
    s.box(245,645,510,65,['FP32 累加各组结果 → yₘₙ'],fill='#eaf5f0',size=22)
    s.text(500,751,'b 是输出通道 n 所在的 128 通道块；正文将 b 写作 b(n)。a、w 是反量化尺度。',18,MUTED)
    s.text(500,786,'V3 的累加提升间隔 N꜀ = 128；实际矩阵更大，量化块与精度策略依据论文图 7。',17,MUTED)
    s.save('fp8-groups.svg')


def ablations():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fonts={f.name for f in font_manager.fontManager.ttflist}
    font=next((x for x in ['Arial Unicode MS','Noto Sans CJK SC','PingFang SC'] if x in fonts),'sans-serif')
    plt.rcParams.update({'font.family':font,'font.size':11,'svg.fonttype':'none','text.color':INK,
                         'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(1,2,figsize=(11,5.6))
    fig.subplots_adjust(left=.12,right=.97,bottom=.24,top=.73,wspace=.37)
    names=['HumanEval','GSM8K','MMLU']
    settings=[
        ([44.5,72.3,67.5],[53.7,74.0,66.6],'MTP 消融 · 表 4 · 540B tokens','基线','加入 MTP'),
        ([40.2,70.7,68.3],[46.3,74.5,67.2],'均衡消融 · 表 5 · 578B tokens','纯辅助损失','无辅助损失')]
    for ax,(a,b,title,l1,l2) in zip(axs,settings):
        y=list(range(3))
        ax.barh([v-.18 for v in y],a,height=.32,label=l1,color=BLUE)
        ax.barh([v+.18 for v in y],b,height=.32,label=l2,color=GREEN)
        for vals,offset in [(a,-.18),(b,.18)]:
            for i,v in enumerate(vals):ax.text(v+1,i+offset,f'{v:.1f}',va='center',fontsize=10)
        ax.set_yticks(y,names);ax.invert_yaxis();ax.set_xlim(0,100)
        ax.set_title(title,fontsize=12,pad=38);ax.set_xlabel('分数（%）')
        ax.legend(loc='lower center',bbox_to_anchor=(.5,1.0),ncol=2,frameon=False,fontsize=10)
        ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('两组独立消融：多数指标改善，也有回落',fontsize=18,y=.97)
    fig.text(.5,.88,'均为 228.7B MoE；两组训练量、基线不同，不能横向相减或叠加收益。',ha='center',fontsize=12,color=MUTED)
    fig.text(.5,.12,'HumanEval：0-shot Pass@1；GSM8K：8-shot EM；MMLU：5-shot EM。',ha='center',fontsize=11,color=MUTED)
    fig.text(.5,.045,'MTP 组在推理时丢弃附加模块；均衡组的双方均使用 sigmoid + Top-K 内归一化。',ha='center',fontsize=11,color=MUTED)
    fig.savefig(OUT/'ablations.svg',facecolor='white');plt.close(fig)


if __name__ == '__main__':
    routing();mtp();dualpipe();fp8();ablations()
