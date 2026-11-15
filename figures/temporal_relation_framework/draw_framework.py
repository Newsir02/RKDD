"""Render editable SVG and 600 dpi PNG diagrams of the proposed method."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


OUTPUT = Path(__file__).resolve().parent
WIDTH, HEIGHT = 1200, 641
INK, MUTED, RULE = "#182536", "#4C5E70", "#C8D3DE"
TEACHER, TEACHER_BG = "#285989", "#EAF2FB"
STUDENT, STUDENT_BG = "#126858", "#E8F5EF"
RELATION, RELATION_BG = "#5D4080", "#F1EBF8"
EVIDENCE_A, EVIDENCE_B = "#A65615", "#9B375F"
PALE = "#F5F8FB"


STRINGS = {
    "en": {
        "title": "Relational distillation of temporal structure interactions",
        "subtitle": "Relational knowledge transfer from a temporal GNN to a snapshot GNN",
        "training": "TRAINING",
        "views": "Four historical views",
        "history": "Observed event prefix",
        "query": "Query",
        "anchor": "Fixed query and candidates",
        "view_labels": ("Neither group", "Keep A", "Keep B", "Keep both"),
        "shared": "Shared edge",
        "masked": "Masked event",
        "models": "Teacher and student relations",
        "teacher": "Frozen temporal GNN",
        "teacher_info": "Historical events and timestamps",
        "student": "Trainable snapshot GNN",
        "student_info": "GCN / GraphSAGE on snapshot adjacency",
        "transfer": "Interaction matching",
        "target": "Fixed target",
        "response": "Student term",
        "match": "Relational loss",
        "total": "Link supervision on the intact view",
        "inference": "INFERENCE",
        "snapshot": "Observed snapshot",
        "snapshot_info": "Intact historical adjacency",
        "deployed": "Trained snapshot GNN",
        "deployed_info": "One forward pass",
        "output": "Candidate link ranking",
        "output_info": "Future link prediction",
    },
    "zh": {
        "title": "时序结构交互关系蒸馏",
        "subtitle": "从时序图神经网络向普通快照图神经网络迁移关系知识",
        "training": "训练阶段",
        "views": "四种历史视图",
        "history": "已观测历史事件",
        "query": "查询",
        "anchor": "固定源节点和候选集",
        "view_labels": ("均遮蔽", "仅保留 A", "仅保留 B", "均保留"),
        "shared": "公共历史边",
        "masked": "遮蔽事件",
        "models": "教师与学生的候选关系",
        "teacher": "冻结的时序 GNN",
        "teacher_info": "历史事件及时间戳",
        "student": "可训练的快照 GNN",
        "student_info": "GCN / GraphSAGE 读取各视图的邻接",
        "transfer": "结构交互关系对齐",
        "target": "冻结教师目标",
        "response": "学生交互响应",
        "match": "关系蒸馏损失",
        "total": "在未遮蔽视图上计算链接监督",
        "inference": "推理阶段",
        "snapshot": "已观测历史快照",
        "snapshot_info": "未遮蔽的历史邻接",
        "deployed": "训练后的快照 GNN",
        "deployed_info": "单次前向传播",
        "output": "候选链接排序",
        "output_info": "未来链接预测",
    },
}


def render(language):
    s = STRINGS[language]
    font = "Microsoft YaHei" if language == "zh" else "Arial"
    plt.rcParams.update({"font.family": font, "mathtext.fontset": "dejavusans",
                         "svg.fonttype": "none", "svg.hashsalt": "relation-framework"})
    # 178 mm is a typical two-column figure width. All font sizes are in points.
    inches = 178 / 25.4
    fig = plt.figure(figsize=(inches, inches * HEIGHT / WIDTH), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, WIDTH), ylim=(HEIGHT, 0), frameon=False)
    ax.set_axis_off()

    def label(x, y, value, size=8, color=INK, bold=False, align="left"):
        return ax.text(x, y, value, fontsize=size, color=color,
                       fontweight="bold" if bold else "normal", ha=align, va="center", zorder=5)

    def line(points, color=RULE, width=.8, dashed=False):
        ax.plot(*zip(*points), color=color, linewidth=width, solid_capstyle="round",
                linestyle=(0, (2, 2)) if dashed else "-", zorder=2)

    def arrow(start, end, color=MUTED, width=.85):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=7,
                                    color=color, linewidth=width, shrinkA=0, shrinkB=0, zorder=3))

    def box(x, y, width, height, fill="white", edge=RULE, radius=9, lw=.7):
        ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle=f"round,pad=0,rounding_size={radius}",
                                    facecolor=fill, edgecolor=edge, linewidth=lw, zorder=1))

    def heading(x, y, index, title):
        label(x, y, f"({index})  {title}", 9.4, bold=True)

    def node(x, y, symbol, is_user, main=False):
        edge = INK if main else MUTED
        fill = "#E1E9F1" if main else "white"
        patch = (Circle((x, y), 11, facecolor=fill, edgecolor=edge, linewidth=.82, zorder=4)
                 if is_user else
                 Rectangle((x - 10.5, y - 10.5), 21, 21, facecolor=fill,
                           edgecolor=edge, linewidth=.82, zorder=4))
        ax.add_patch(patch)
        label(x, y, f"${symbol}$", 7.9, align="center")

    def view_tile(x, y, i, a, b):
        box(x, y, 159, 91, PALE if a and b else "white")
        label(x + 10, y + 17, f"$H_{{{a}{b}}}$", 9.0, bold=True)
        label(x + 149, y + 17, s["view_labels"][i], 6.4, MUTED, align="right")
        xs = [x + offset for offset in (19, 59, 100, 140)]
        yc = y + 65
        line([(xs[0], yc), (xs[1], yc)], EVIDENCE_A if a else RULE, 1.5 if a else .95, not a)
        line([(xs[1], yc), (xs[2], yc)], MUTED, 1.0)
        line([(xs[2], yc), (xs[3], yc)], EVIDENCE_B if b else RULE, 1.5 if b else .95, not b)
        label((xs[0] + xs[1]) / 2, yc - 17, "$A$", 7.1, EVIDENCE_A if a else MUTED, align="center")
        label((xs[2] + xs[3]) / 2, yc - 17, "$B$", 7.1, EVIDENCE_B if b else MUTED, align="center")
        for xp, symbol, is_user in zip(xs, ("u", "a", "b", "v"), (True, False, True, False)):
            node(xp, yc, symbol, is_user, symbol in ("u", "v"))

    def relation_row(y, key, color, tint, index):
        box(420, y, 375, 122, tint, color)
        label(436, y + 22, s[key], 9.4, color, bold=True)
        label(436, y + 48, s[f"{key}_info"], 7.25, MUTED)
        for j, variant in enumerate(("00", "10", "01", "11")):
            x = 436 + j * 85
            box(x, y + 70, 77, 34, "white", color, radius=5, lw=.55)
            label(x + 38.5, y + 87, rf"$R_{{{index}}}^{{{variant}}}$", 10, color, align="center")

    label(24, 28, s["title"], 15, bold=True)
    label(24, 59, s["subtitle"], 8.6, MUTED)
    line([(24, 79), (1176, 79)], INK, 1.0)
    label(24, 101, s["training"], 7.6, MUTED, bold=True)

    heading(24, 129, "a", s["views"])
    heading(420, 129, "b", s["models"])
    heading(836, 129, "c", s["transfer"])

    label(25, 161, s["history"], 7.3, MUTED)
    label(324, 161, rf"{s['query']} $t$", 7.3, MUTED, align="center")
    arrow((32, 187), (351, 187), MUTED, .75)
    for x in (59, 154, 218, 261):
        ax.add_patch(Circle((x, 187), 2.8, color=RULE, zorder=4))
    for x, text, color in [(105, "$A$", EVIDENCE_A), (191, "$B$", EVIDENCE_B)]:
        ax.add_patch(Circle((x, 187), 3.9, color=color, zorder=4))
        label(x, 202, text, 7.4, color, align="center")
    line([(324, 175), (324, 196)], INK, .9, dashed=True)
    label(24, 225, s["anchor"], 7.25, MUTED)
    label(353, 225, r"$(u;v,w)$", 8.1, align="right")
    for i, (a, b) in enumerate(((0, 0), (1, 0), (0, 1), (1, 1))):
        view_tile(24 + (i % 2) * 170, 244 + (i // 2) * 104, i, a, b)

    line([(30, 460), (52, 460)], MUTED, 1.0)
    label(59, 460, s["shared"], 7.1, MUTED)
    line([(200, 460), (222, 460)], RULE, .95, dashed=True)
    label(229, 460, s["masked"], 7.1, MUTED)
    # Collect the four views into a shared pair of model inputs.
    line([(362, 245), (370, 245), (370, 439), (362, 439)], MUTED, .8)
    line([(370, 341), (396, 341)], MUTED, .9)
    line([(396, 244), (396, 404)], MUTED, .9)
    arrow((396, 244), (415, 244), TEACHER)
    arrow((396, 404), (415, 404), STUDENT)

    label(607.5, 162, r"$R_M^{ij}=z_M(u,v\,|\,H_{ij})-z_M(u,w\,|\,H_{ij})$", 9.15, align="center")
    relation_row(183, "teacher", TEACHER, TEACHER_BG, "T")
    relation_row(343, "student", STUDENT, STUDENT_BG, "S")

    label(1005, 162, r"$I_M=R_M^{11}-R_M^{10}-R_M^{01}+R_M^{00}$", 9.25, RELATION, align="center")
    for y, index, name, color, tint in [(244, "T", "target", TEACHER, TEACHER_BG),
                                         (404, "S", "response", STUDENT, STUDENT_BG)]:
        arrow((799, y), (831, y), color)
        box(836, y - 28, 123, 56, tint, color)
        label(897.5, y - 10, rf"$I_{{{index}}}$", 12.5, color, align="center")
        label(897.5, y + 13, s[name], 7.0, color, align="center")
    line([(962, 244), (977, 244), (977, 306)], TEACHER)
    arrow((977, 306), (991, 306), TEACHER)
    line([(962, 404), (977, 404), (977, 344)], STUDENT)
    arrow((977, 344), (991, 344), STUDENT)
    box(996, 286, 180, 78, RELATION_BG, RELATION, lw=1.05)
    label(1086, 305, s["match"], 8.1, RELATION, bold=True, align="center")
    label(1086, 338, r"$\mathrm{Huber}(I_S,I_T)$", 10.4, RELATION, align="center")
    label(1176, 460, r"$\mathcal{L}=\mathcal{L}_{\mathrm{link}}+\lambda\mathcal{L}_{\mathrm{rel}}$",
          9.6, align="right")
    label(1176, 481, s["total"], 7.1, MUTED, align="right")

    line([(24, 507), (1176, 507)], INK, .9)
    label(24, 532, s["inference"], 7.6, MUTED, bold=True)
    items = [(24, 346, "snapshot", PALE, RULE, INK),
             (420, 375, "deployed", STUDENT_BG, STUDENT, STUDENT),
             (836, 340, "output", PALE, RULE, INK)]
    for x, width, key, fill, edge, color in items:
        box(x, 552, width, 66, fill, edge)
        label(x + width / 2, 572, s[key], 9.2, color, bold=True, align="center")
        label(x + width / 2, 598, s[f"{key}_info"], 7.2, MUTED, align="center")
    arrow((375, 585), (415, 585))
    arrow((800, 585), (831, 585), STUDENT)

    filename = OUTPUT / f"temporal_relation_framework_{language}"
    fig.savefig(filename.with_suffix(".svg"), metadata={"Date": None, "Title": s["title"]})
    fig.savefig(filename.with_suffix(".png"), dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    for locale in ("zh", "en"):
        render(locale)
