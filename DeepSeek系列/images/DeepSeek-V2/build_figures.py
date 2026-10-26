"""Rebuild the four authored SVGs with `python3 build_figures.py`.

Requires matplotlib and a Chinese font. Render the two .mmd sources with
Mermaid 10.9.3 to their same-name .svg files. Paper: arXiv:2405.04434v5.
Diagrams are original teaching illustrations; the chart uses Appendix D Table 9.
"""
from html import escape
from pathlib import Path

OUT = Path(__file__).resolve().parent
INK, MUTED = '#183049', '#546778'
BLUE, GREEN, ORANGE = '#356da8', '#287c64', '#ba6831'


class SVG:
    def __init__(self, w, h, title):
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img">',
                      f'<title>{escape(title)}</title>',
                      '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8 Z" fill="#546778"/></marker></defs>',
                      f'<rect width="{w}" height="{h}" fill="white"/>']

    def text(self, x, y, value, size=20, color=INK, anchor='middle', weight=400):
        self.parts.append(f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" fill="{color}" font-weight="{weight}" font-family="Arial Unicode MS, Noto Sans CJK SC, PingFang SC, sans-serif">{escape(value)}</text>')

    def box(self, x, y, w, h, lines, fill='#eef4fb', size=20, stroke='#c8d4df'):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" fill="{fill}" stroke="{stroke}"/>')
        start = y+h/2-(len(lines)-1)*size*1.45/2+size*.35
        for i, v in enumerate(lines):
            self.text(x+w/2, start+i*size*1.45, v, size)

    def line(self, pts, arrow=True, dashed=False):
        points = ' '.join(f'{x},{y}' for x, y in pts)
        end = ' marker-end="url(#arrow)"' if arrow else ''
        dash = ' stroke-dasharray="6 5"' if dashed else ''
        self.parts.append(f'<polyline points="{points}" fill="none" stroke="{MUTED}" stroke-width="2"{end}{dash}/>')

    def save(self, name):
        (OUT/name).write_text('\n'.join(self.parts+['</svg>']), encoding='utf-8')


def compression():
    s = SVG(1000, 730, 'MLA 联合压缩的矩阵形状；列向量，单 token')
    s.text(500, 38, '一份 KV 潜在表示，供 128 个头分别读取', 26, weight=600)
    s.text(500, 70, '列向量约定 · 本图只画内容分支 · 省略归一化与额外缩放', 18, MUTED)
    s.box(350, 96, 300, 66, ['输入 hₜ：[5120, 1]'])
    s.line([(430,162),(430,185),(250,185),(250,210)])
    s.line([(570,162),(570,185),(750,185),(750,210)])
    s.box(40,210,420,80,['KV 降维 Wᴰᴷⱽ：[512, 5120]','cₜᴷⱽ：[512, 1]'],fill='#eaf5f0',stroke=GREEN)
    s.box(540,210,420,80,['Q 降维 WᴰQ：[1536, 5120]','cₜQ：[1536, 1]'])
    s.line([(70,290),(70,365)])
    s.line([(570,290),(570,365)])
    s.text(250,328,'这一份潜在表示进入历史缓存',19,GREEN)
    s.text(750,328,'当前查询使用；不缓存历史 Q',19,BLUE)
    s.box(40,365,420,90,['每头的 Wᵁᴷᵢ、Wᵁⱽᵢ','各为 [128, 512]'])
    s.box(540,365,420,90,['每头的 WᵁQᵢ','[128, 1536]'])
    s.line([(250,455),(250,490)])
    s.line([(750,455),(750,490)])
    s.box(40,490,420,80,['单头 kᶜₜᵢ、vₜᵢ：各 [128, 1]','所有头堆叠：各 [16384, 1]'])
    s.box(540,490,420,80,['单头 qᶜₜᵢ：[128, 1]','所有头堆叠：[16384, 1]'])
    s.box(40,610,920,70,['联合线性投影：[32768, 512] × [512, 5120]','秩 ≤ 512；共享的是 cᴷⱽ，各头的读取矩阵仍不同'],fill='#f5f7fa')
    s.text(500,710,'完整缓存还包含 64 维位置 K（见解耦 RoPE 图）；解码时可用矩阵吸收跳过历史 K/V 展开。',17,MUTED)
    s.save('mla-compression.svg')


def absorption():
    s = SVG(1000, 760, 'MLA 矩阵吸收：保持结果，改变历史状态参与计算的方式')
    s.text(500,38,'矩阵吸收：把计算移到查询侧或加权汇总之后',25,weight=600)
    s.text(500,71,'列向量 · 单个头 i · cⱼ = cⱼᴷⱽ · q = qᶜₜᵢ · S 个历史位置',18,MUTED)
    s.text(30,112,'① K 侧：点积前，先变换谁？',22,anchor='start',weight=600)
    s.box(30,140,430,130,['逐位置展开历史 K','cⱼ [512,1] → Wᵁᴷᵢ → kᶜⱼᵢ [128,1]','再计算 qᵀ kᶜⱼᵢ'],fill='#f5f7fa',size=20)
    s.box(540,140,430,130,['只变换当前 Q','q [128,1] → (Wᵁᴷᵢ)ᵀ → q̃ [512,1]','再计算 q̃ᵀ cⱼ'],fill='#eaf5f0',size=20)
    s.line([(470,205),(530,205)])
    s.text(500,303,'qᵀ (Wᵁᴷᵢ cⱼ) = ((Wᵁᴷᵢ)ᵀ q)ᵀ cⱼ',24)
    s.text(500,337,'C：[512,S]；内容分数 q̃ᵀC：[1,S]；完整权重转为列向量 a：[S,1]',19,MUTED)
    s.text(30,390,'② V 侧：先展开，还是先汇总？',22,anchor='start',weight=600)
    s.box(30,420,430,145,['逐位置展开历史 V','cⱼ → Wᵁⱽᵢ → vⱼᵢ [128,1]','按 aⱼ 汇总 → Wᴼᵢ → 输出 [5120,1]'],fill='#f5f7fa',size=19)
    s.box(540,420,430,145,['先汇总潜在表示','z = C a：[512,1]','再经 Wᴼᵢ Wᵁⱽᵢ → 输出 [5120,1]'],fill='#eaf5f0',size=19)
    s.line([(470,492),(530,492)])
    s.text(500,602,'Wᴼᵢ Σⱼ aⱼ (Wᵁⱽᵢ cⱼ) = (Wᴼᵢ Wᵁⱽᵢ) Σⱼ aⱼ cⱼ',23)
    s.box(30,635,940,72,['不必物化 [128,S] 的历史 K、V；每个头仍有自己的 a 与 z。','可合并权重，也可分步计算；代数等价不代表 FLOPs 按缓存比例下降。'],size=19)
    s.text(500,739,'此图只比较内容分支；完整打分还需加入位置分支，并使用原始 Q/K 维度的缩放。',17,MUTED)
    s.save('mla-absorption.svg')


def rope():
    s = SVG(1000, 825, '解耦 RoPE：内容与位置分别匹配，共用一个 softmax')
    s.text(500,38,'解耦 RoPE：内容分支保留吸收，位置分支独立旋转',25,weight=600)
    s.box(30,65,940,90,['若直接旋转内容：qᵀ [Rₜᵀ Rⱼ] Wᵁᴷᵢ cⱼ','中间旋转随历史位置 j 改变，无法预先合成一个通用查询投影。'],fill='#fff2e7',size=21)
    s.text(250,195,'内容匹配（蓝色）',22,BLUE,weight=600)
    s.text(750,195,'位置匹配（橙色）',22,ORANGE,weight=600)
    s.box(30,220,440,100,['当前 cₜQ：[1536,1]','经 (Wᵁᴷᵢ)ᵀ WᵁQᵢ → q̃ₜᵢ：[512,1]'],size=20)
    s.box(530,220,440,100,['当前 cₜQ：[1536,1]','WQᴿᵢ：[64,1536] 投影，再由 Rₜ 旋转','得到每头 qᴿₜᵢ：[64,1]'],fill='#fff2e7',size=19)
    s.line([(140,320),(140,435)])
    s.line([(640,320),(640,435)])
    s.box(220,345,250,65,['历史 cⱼᴷⱽ：[512,1]','所有头共享缓存'],fill='#eaf5f0',size=18,stroke=GREEN)
    s.box(720,345,250,65,['历史 kᴿⱼ：[64,1]','所有头共享缓存'],fill='#eaf5f0',size=18,stroke=GREEN)
    s.line([(345,410),(345,435)])
    s.line([(845,410),(845,435)])
    s.box(30,435,440,65,['点积 q̃ₜᵢᵀ cⱼᴷⱽ → 标量'],size=21)
    s.box(530,435,440,65,['点积 (qᴿₜᵢ)ᵀ kᴿⱼ → 标量'],fill='#fff2e7',size=21)
    s.line([(250,500),(250,530),(430,530),(430,555)])
    s.line([(750,500),(750,530),(570,530),(570,555)])
    s.box(255,555,490,85,['两个标量相加，再除以 √192','对历史位置 j ≤ t 做一次 softmax'],fill='#f5f7fa',size=22)
    s.line([(500,640),(500,670)])
    s.box(255,670,490,60,['输出每头权重 aₜⱼᵢ → 加权汇总 V'],size=21)
    s.text(500,766,'位置 K 的来源：hⱼ [5120,1] → Wᴷᴿ [64,5120] → Rⱼ → kᴿⱼ [64,1]',19,ORANGE)
    s.text(500,800,'每层每 token 缓存：512 + 64 = 576 个元素；绿色框是缓存，Q 不进入历史缓存。',18,GREEN)
    s.save('mla-rope.svg')


def ablation():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fonts = {f.name for f in font_manager.fontManager.ttflist}
    font = next((f for f in ['Arial Unicode MS','Noto Sans CJK SC','PingFang SC'] if f in fonts), 'sans-serif')
    plt.rcParams.update({'font.family':font,'font.size':12,'svg.fonttype':'none',
                         'text.color':INK,'axes.labelcolor':INK,'axes.spines.top':False,'axes.spines.right':False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    fig.subplots_adjust(left=.1,right=.97,bottom=.23,top=.76,wspace=.40)
    labels = ['MHA','MLA']
    for ax, vals, title, xmax in zip(axes,[[860.2,34.6],[57.5,59.0]],
                                    ['每 token 的全模型 KV 缓存（千元素）','MMLU 准确率（%，5-shot）'],[1030,100]):
        ax.barh(labels,vals,color=[BLUE,GREEN],height=.5)
        ax.invert_yaxis();ax.set_xlim(0,xmax);ax.set_title(title,fontsize=13,pad=16)
        ax.spines[['left']].set_visible(False);ax.tick_params(axis='y',length=0)
        ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
        for y,v in enumerate(vals): ax.text(v+xmax*.02,y,f'{v:.1f}',va='center',fontsize=13)
    fig.suptitle('注意力消融：缓存大幅减少，MMLU 略有提高',fontsize=18,x=.53,y=.97)
    fig.text(.5,.86,'大规模 MoE 对照 · 均训练 420B tokens · 除注意力外结构相同',ha='center',fontsize=12,color=MUTED)
    fig.text(.5,.10,'总参数：250.8B → 247.4B；激活参数：25.0B → 21.5B。参数量并非完全相同。',ha='center',fontsize=11,color=MUTED)
    fig.text(.5,.035,'来源：论文附录 D.2，表 9；本图重新绘制，仅展示其中两项指标。',ha='center',fontsize=11,color=MUTED)
    fig.savefig(OUT/'mla-ablation.svg',facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    compression(); absorption(); rope(); ablation()
