"""Publication-style framework for temporal structural interaction distillation."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


OUT = Path(__file__).resolve().parent
W, H = 1280, 551
INK, TEXT, GUIDE = "#1F2B37", "#495969", "#CFD8DF"
A, B = "#AF552E", "#16777F"
T, T_PALE = "#305D8B", "#EDF3FA"
S, S_PALE = "#386C51", "#EDF5EF"
R, R_PALE = "#5D4481", "#F2EDF8"
POS, NEG = "#664992", "#965236"
CELLS = ((0, 0), (0, 1), (1, 0), (1, 1))

LABELS = {
    "zh": {
        "views": "历史结构干预", "relations": "师生关系响应", "match": "结构交互蒸馏",
        "prefix": "已观测事件", "cut": "查询时刻", "query": "固定源节点与候选",
        "common": "公共历史边", "fixed": "其余历史保持不变",
        "teacher": "冻结时序 GNN", "student": "快照 GNN 学生",
        "time": "事件与时间戳", "adj": "历史邻接", "model": "GCN / GraphSAGE",
        "signs": "共享差分算子", "kd": "关系对齐", "supervision": "链接监督来自未遮蔽视图",
        "infer": "推理", "full": "未遮蔽历史快照", "trained": "训练后的学生 GNN",
        "ranking": "候选链接排序",
    },
    "en": {
        "views": "Historical interventions", "relations": "Teacher–student responses",
        "match": "Interaction distillation",
        "prefix": "Observed events", "cut": "Query time", "query": "Fixed source and candidates",
        "common": "Shared edge", "fixed": "All other history is held fixed",
        "teacher": "Frozen temporal GNN", "student": "Snapshot GNN student",
        "time": "Timestamped events", "adj": "Graph adjacency", "model": "GCN / GraphSAGE",
        "signs": "Shared contrast", "kd": "Match relations", "supervision": "Link loss on the intact view",
        "infer": "Inference", "full": "Intact snapshot", "trained": "Trained student GNN",
        "ranking": "Candidate link ranking",
    },
}


def render(locale, output, preview=False):
    s = LABELS[locale]
    font = "Microsoft YaHei" if locale == "zh" else "Arial"
    plt.rcParams.update({"font.family": font, "mathtext.fontset": "stix",
                         "svg.fonttype": "none", "svg.hashsalt": "interaction-framework-v2"})
    width_in = 178 / 25.4
    fig = plt.figure(figsize=(width_in, width_in * H / W), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, W), ylim=(H, 0))
    ax.set_axis_off()

    def txt(x, y, value, size=7.5, color=INK, bold=False, align="left"):
        ax.text(x, y, value, color=color, fontsize=size, fontweight="bold" if bold else "normal",
                ha=align, va="center", zorder=6)

    def path(points, color=GUIDE, width=.8, style="-"):
        ax.plot(*zip(*points), color=color, linewidth=width, linestyle=style,
                solid_capstyle="round", zorder=3)

    def arrow(start, end, color=TEXT, width=.85, style="-"):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=7.6,
                                    linewidth=width, linestyle=style, color=color,
                                    shrinkA=0, shrinkB=0, zorder=4))

    def rect(x, y, w, h, fill="white", edge="none", radius=5, width=.7):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
                                    facecolor=fill, edgecolor=edge, linewidth=width, zorder=1))

    def circle(x, y, radius, fill, edge="none", width=.8):
        ax.add_patch(Circle((x, y), radius, facecolor=fill, edgecolor=edge,
                            linewidth=width, zorder=5))

    def node(x, y, symbol, account, endpoint=False):
        fill = "#E6EEF4" if endpoint else "white"
        if account:
            circle(x, y, 10.6, fill, INK)
        else:
            ax.add_patch(Rectangle((x - 10, y - 10), 20, 20, facecolor=fill,
                                   edgecolor=INK, linewidth=.8, zorder=5))
        txt(x, y, "$" + symbol + "$", 8.5, align="center")

    def section(x, mark, label):
        txt(x, 29, f"({mark})", 9.15, bold=True)
        txt(x + 46, 29, label, 9.75, bold=True)

    section(20, "a", s["views"])
    section(419, "b", s["relations"])
    section(856, "c", s["match"])
    path([(20, 47), (375, 47)])
    path([(419, 47), (815, 47)])
    path([(856, 47), (1260, 47)])

    # All structural interventions share the same historical cutoff.
    txt(21, 71, s["prefix"], 7.3, TEXT)
    txt(328, 71, rf"{s['cut']} $t$", 7.3, TEXT, align="center")
    arrow((28, 98), (368, 98), TEXT, .72)
    for x in (45, 107, 169, 210, 258, 297):
        circle(x, 98, 3, "#BAC8D3")
    for x, symbol, color in [(82, "A", A), (230, "B", B)]:
        circle(x, 98, 4.6, color)
        txt(x, 115, "$" + symbol + "$", 8.3, color, align="center")
    path([(328, 86), (328, 111)], TEXT, .85, "--")
    txt(21, 138, s["query"], 7.2, TEXT)
    txt(367, 138, r"$(u;v,w)$", 8.6, align="right")

    # Row i enables A. Column j enables B. The relation matrices use that order.
    grid_x, grid_y, cell_w, cell_h = 48, 184, 156, 109
    for j in (0, 1):
        txt(grid_x + (j + .5) * cell_w, 165, f"$B={j}$", 8.3, B, bold=True, align="center")
    for i in (0, 1):
        txt(9, grid_y + (i + .5) * cell_h, f"$A={i}$", 7.6, A, bold=True)
    for i, j in CELLS:
        x, y = grid_x + j * cell_w, grid_y + i * cell_h
        rect(x + 3, y + 3, cell_w - 6, cell_h - 6,
             "#EEF5F5" if i and j else "#F7F9FB", radius=4)
        txt(x + 13, y + 22, rf"$H_{{{i}{j}}}$", 9.3)
        points = [(x + 21, y + 81), (x + 58, y + 50),
                  (x + 97, y + 81), (x + 134, y + 50)]
        path(points[1:3], "#8795A0", 1.05)
        if i:
            path(points[0:2], A, 1.75)
        if j:
            path(points[2:4], B, 1.75)
        for n, (xp, yp) in enumerate(points):
            node(xp, yp, "uabv"[n], n % 2 == 0, n in (0, 3))
    path([(51, 426), (76, 426)], "#8795A0", 1.05)
    txt(85, 426, s["common"], 7.1, TEXT)
    txt(51, 453, s["fixed"], 7.1, TEXT)
    path([(367, 185), (375, 185), (375, 400), (367, 400)], "#768694", .85)
    path([(375, 291), (397, 291)], "#768694", .9)
    path([(397, 175), (397, 385)], "#768694", .9)
    arrow((397, 175), (416, 175), T)
    arrow((397, 385), (416, 385), S)

    txt(617, 74, r"$R_M^{ij}=z_M(u,v\,|\,H_{ij})-z_M(u,w\,|\,H_{ij})$",
        9.4, align="center")

    def event_glyph(x, y):
        arrow((x, y), (x + 112, y), T, .7)
        for xp, color in [(x + 12, A), (x + 42, T), (x + 74, B), (x + 96, T)]:
            circle(xp, y, 4.4, color)
            path([(xp, y - 4), (xp, y - 17)], color, .65)
        txt(x + 56, y + 26, s["time"], 7.25, TEXT, align="center")

    def graph_glyph(x, y):
        nodes = [(x + 8, y + 5), (x + 39, y - 19), (x + 73, y + 9), (x + 105, y - 10)]
        path(nodes, S, 1.1)
        for n, (xp, yp) in enumerate(nodes):
            circle(xp, yp, 5.2, "white" if n in (1, 2) else "#B8D8C4", S)
        txt(x + 56, y + 26, s["adj"], 7.25, TEXT, align="center")

    def matrix(x, y, content, fills, color, cell_w=61, cell_h=38, size=9.6, text_colors=None):
        for n, (i, j) in enumerate(CELLS):
            left, top = x + j * cell_w, y + i * cell_h
            rect(left + 1.5, top + 1.5, cell_w - 3, cell_h - 3, fills[n], radius=2)
            txt(left + cell_w / 2, top + cell_h / 2, content[n], size,
                color if text_colors is None else text_colors[n], align="center")
        end_x, end_y = x + 2 * cell_w, y + 2 * cell_h
        path([(x-1, y-2), (x-6, y-2), (x-6, end_y+2), (x-1, end_y+2)], color, .9)
        path([(end_x+1, y-2), (end_x+6, y-2), (end_x+6, end_y+2), (end_x+1, end_y+2)], color, .9)

    for center, key, index, color, tint, glyph in [
        (175, "teacher", "T", T, T_PALE, event_glyph),
        (385, "student", "S", S, S_PALE, graph_glyph),
    ]:
        rect(419, center - 60, 396, 120, tint, radius=8)
        txt(436, center - 39, s[key], 8.9, color, True)
        if index == "S":
            txt(436, center - 17, s["model"], 7.1, TEXT)
        glyph(451, center + 18)
        arrow((586, center + 13), (632, center + 13), color)
        matrix(651, center - 15, [rf"$R_{index}^{{{i}{j}}}$" for i, j in CELLS],
               ["white"] * 4, color, cell_w=69, cell_h=32)
        arrow((819, center), (873, center), color)

    # This contraction is exactly the difference of differences.
    txt(1051, 73, r"$I_M=\langle \mathbf{W},\mathbf{R}_M\rangle_F$", 11.5, R, align="center")
    txt(1051, 99, r"$=R_M^{11}-R_M^{10}-R_M^{01}+R_M^{00}$",
        9.2, R, align="center")

    def contract(y, color, index):
        ax.add_patch(Polygon([(895, y-17), (913, y), (895, y+17), (877, y)],
                             closed=True, facecolor=R_PALE, edgecolor=R, linewidth=.92, zorder=5))
        txt(895, y, r"$\Sigma$", 9, R, align="center")
        arrow((919, y), (1036, y), color)
        rect(1043, y - 21, 79, 42, "white", color, radius=19, width=.9)
        txt(1082.5, y, rf"$I_{index}$", 12.8, color, align="center")

    contract(175, T, "T")
    contract(385, S, "S")
    path([(895, 196), (895, 262), (903, 262)], R, .67, ":")
    path([(895, 364), (895, 308), (903, 308)], R, .67, ":")
    txt(915, 244, s["signs"], 7.2, TEXT)
    txt(867, 286, r"$\mathbf{W}=$", 8.7, R)
    matrix(914, 254, [r"$+1$", r"$-1$", r"$-1$", r"$+1$"],
           ["#F0E9F9", "#FAEDE6", "#FAEDE6", "#F0E9F9"], R,
           cell_w=40, cell_h=30, size=9.1, text_colors=(POS, NEG, NEG, POS))
    arrow((1082.5, 205), (1082.5, 356), R, 1.5, "--")
    txt(1100, 265, s["kd"], 8.0, R, True)
    txt(1100, 296, r"$\mathrm{Huber}(I_S,I_T)$", 9.0, R)
    txt(1260, 432, r"$\mathcal{L}=\mathcal{L}_{\mathrm{link}}+\lambda\mathcal{L}_{\mathrm{rel}}$",
        10.25, align="right")
    txt(1260, 457, s["supervision"] + r" $H_{11}$", 7.4, TEXT, align="right")

    path([(20, 481), (1260, 481)], GUIDE, .85)
    txt(20, 516, s["infer"], 8.2, INK, True)
    txt(218, 516, s["full"], 8.0, TEXT, align="center")
    arrow((345, 516), (401, 516), TEXT, .75)
    rect(419, 497, 396, 39, S_PALE, radius=5)
    txt(617, 516, s["trained"], 8.45, S, True, align="center")
    arrow((833, 516), (890, 516), S, .8)
    txt(1034, 516, s["ranking"], 8.3, INK, align="center")

    output.mkdir(parents=True, exist_ok=True)
    name = output / f"temporal_relation_framework_v2_{locale}"
    if not preview:
        fig.savefig(name.with_suffix(".svg"), metadata={"Date": None})
    fig.savefig(name.with_suffix(".png"), dpi=150 if preview else 600)
    plt.close(fig)


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--output", type=Path, default=OUT)
    cli.add_argument("--preview", action="store_true")
    args = cli.parse_args()
    for language in ("zh", "en"):
        render(language, args.output, args.preview)
