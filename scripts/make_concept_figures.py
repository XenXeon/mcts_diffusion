"""scripts/make_concept_figures.py

Generate the five CONCEPTUAL diagrams for the methods chapter (dissertation
Chapter 3) into figures/ as vector PDF (for Word/LaTeX) + PNG preview.

Unlike scripts/make_figures.py these carry no experimental data — they are
schematics of the systems and algorithms specified in Chapter 3. They are scripted
rather than hand-drawn so that they regenerate identically and stay consistent with
the results figures' palette.

Palette: identical to make_figures.py (validated dataviz defaults). Categorical
slots #2a78d6/#1baf7a/#e34948 validated via the dataviz validator: CVD separation
for the red/aqua pair sits in the 6-8 floor band, which is legal only with
secondary encoding -- every coloured element in these figures is DIRECTLY LABELLED
in text, so identity is never carried by colour alone. Noise level is a magnitude,
so it uses a single-hue sequential blue ramp (light = clean, dark = noisy), never a
categorical or rainbow scale.

Run (needs matplotlib; torch-free):  python scripts/make_concept_figures.py
"""
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.colors import LinearSegmentedColormap

# ── palette (identical to make_figures.py) ─────────────────────────────────
BLUE, AQUA, RED, ORANGE = "#2a78d6", "#1baf7a", "#e34948", "#eb6834"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASE = "#e1e0d9", "#c3c2b7"
GOOD, CRIT = "#006300", "#d03b3b"
SURF = "#ffffff"

# sequential single-hue ramp for NOISE LEVEL (light = clean, dark = noisy)
NOISE_CMAP = LinearSegmentedColormap.from_list(
    "noise", ["#eaf2fc", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

OUT = "figures"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Segoe UI", "Arial"],
    "font.size": 9.5,
    "axes.edgecolor": BASE, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "axes.labelcolor": INK2, "axes.labelsize": 9.5,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "figure.facecolor": SURF, "axes.facecolor": SURF,
    "savefig.facecolor": SURF, "savefig.bbox": "tight", "savefig.dpi": 200,
})


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  wrote {OUT}/{name}.pdf (+.png)")


def box(ax, x, y, w, h, label, sub=None, fc="#f4f7fb", ec=BLUE, lw=1.6, fs=9.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.012,rounding_size=0.02",
                                fc=fc, ec=ec, lw=lw, zorder=3))
    ax.text(x + w / 2, y + h / 2 + (0.035 if sub else 0), label, ha="center",
            va="center", fontsize=fs, color=INK, fontweight="bold", zorder=4)
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.045, sub, ha="center", va="center",
                fontsize=7.8, color=INK2, zorder=4)


def arrow(ax, p0, p1, label=None, color=INK2, style="-|>", lw=1.4,
          rad=0.0, lab_off=(0, 0.03), fs=8.2):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=12,
                                 lw=lw, color=color,
                                 connectionstyle=f"arc3,rad={rad}", zorder=2))
    if label:
        mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
        ax.text(mx + lab_off[0], my + lab_off[1], label, ha="center", va="bottom",
                fontsize=fs, color=MUTED, zorder=4)


def blank(ax):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")


def cosine_abar(K=20, s=0.008):
    """Nichol & Dhariwal cosine schedule, abar[0] = 1 exactly."""
    k = np.arange(K + 1)
    f = np.cos(((k / K + s) / (1 + s)) * np.pi / 2) ** 2
    return f / f[0]


# ════════════════════════════════════════════════════════════════════════
# Figure 3-1 — the frozen three-network system and the MPC loop
# ════════════════════════════════════════════════════════════════════════
def fig_system():
    fig, ax = plt.subplots(figsize=(8.2, 4.5)); blank(ax)
    YB, HB = 0.50, 0.155                                   # box band

    box(ax, 0.020, YB, 0.150, HB, "Environment", "state $s_t$", fc="#f7f7f5", ec=BASE)
    box(ax, 0.245, YB, 0.170, HB, "Planner", "diffusion, frozen", ec=BLUE)
    box(ax, 0.490, YB, 0.150, HB, "Critic", "frozen", ec=AQUA)
    box(ax, 0.715, YB, 0.170, HB, "Inverse\ndynamics", "frozen", ec=RED)

    yc, ytop = YB + HB / 2, YB + HB
    for x0, x1, lab in [(0.175, 0.240, None), (0.420, 0.485, "K windows"),
                        (0.645, 0.710, "argmax"), (0.890, 0.945, None)]:
        arrow(ax, (x0, yc), (x1, yc))
        if lab:
            ax.text((x0 + x1) / 2, ytop + 0.022, lab, ha="center", va="bottom",
                    fontsize=8.0, color=MUTED)
    ax.text(0.955, yc, "$a_t$", ha="left", va="center", fontsize=10.5, color=INK)

    # candidate windows, stacked under the planner
    for dy in (0.435, 0.398, 0.361):
        for r in range(7):
            ax.add_patch(Rectangle((0.253 + r * 0.0145, dy), 0.0125, 0.027,
                                   fc=NOISE_CMAP(0.16), ec="white", lw=0.4, zorder=3))
    ax.text(0.372, 0.412, "K candidate plan windows\n(H rows x D dims)", ha="left",
            va="center", fontsize=7.8, color=MUTED)

    # feedback loop, routed explicitly so it cannot clip
    YL = 0.265
    ax.plot([0.968, 0.968], [YB - 0.005, YL], color=MUTED, lw=1.2, zorder=2)
    ax.plot([0.095, 0.968], [YL, YL], color=MUTED, lw=1.2, zorder=2)
    ax.add_patch(FancyArrowPatch((0.095, YL), (0.095, YB - 0.005),
                                 arrowstyle="-|>", mutation_scale=12, lw=1.2,
                                 color=MUTED, zorder=2))
    ax.text(0.53, YL - 0.055,
            "execute one action, observe the true next state, replan\n"
            "(model-predictive control)",
            ha="center", va="top", fontsize=8.4, color=MUTED, style="italic")

    ax.text(0.02, 0.955, "The inference rule is the only thing this dissertation varies",
            ha="left", va="top", fontsize=10.5, color=INK, fontweight="bold")
    ax.text(0.02, 0.855,
            "All three networks are trained independently and held frozen: a difference between two arms\n"
            "cannot come from what the system knows, only from how it deploys that knowledge.",
            ha="left", va="top", fontsize=8.4, color=INK2)
    save(fig, "fig3_1_system_mpc")


# ════════════════════════════════════════════════════════════════════════
# Figure 3-2 — the three expansion mechanisms (the central comparison)
# ════════════════════════════════════════════════════════════════════════
def fig_expansion():
    H, PREFIX = 12, 4
    fig, axes = plt.subplots(1, 3, figsize=(9.4, 4.9))
    fig.subplots_adjust(wspace=0.55)

    panels = [
        ("(a)  Seam-glue", "concatenate", RED,
         "continuation sampled from the\nleaf STATE ALONE, then\nconcatenated onto the prefix",
         "seam: this junction was\nnever generated\nby the model"),
        ("(b)  Replacement\n      inpainting", "clamp", RED,
         "prefix rows clamped into the\ndenoiser at every step;\nno seam remains",
         "mixed noise levels in one\ninput -- a configuration\nnever seen in training"),
        ("(c)  Exact prefix\n      conditioning", "condition", GOOD,
         "prefix held at level 0 while\nthe rest denoises; a true\nconditional sample",
         "in-distribution by\nconstruction: the model\nwas trained on this"),
    ]

    for pi, (ax, (title, _, flag, how, note)) in enumerate(zip(axes, panels)):
        ax.set_xlim(-0.75, 1.55); ax.set_ylim(-6.2, H + 2.4); ax.axis("off")
        ax.text(0.45, H + 1.9, title, ha="center", va="top", fontsize=9.6,
                color=INK, fontweight="bold", linespacing=1.35)

        for r in range(H):
            y = H - 1 - r
            if r < PREFIX:
                lvl = 0.0
            elif title.startswith("(a)"):
                lvl = 0.10                       # independently denoised, already clean
            elif title.startswith("(b)"):
                lvl = 0.55                       # uniformly mid-noise around clamped rows
            else:
                lvl = 0.12 + 0.78 * (r - PREFIX) / max(1, H - PREFIX - 1)
            ax.add_patch(Rectangle((0, y), 0.9, 0.82, fc=NOISE_CMAP(lvl),
                                   ec="white", lw=0.7, zorder=3))

        if pi == 0:                                  # row-group labels once only
            ax.annotate("", xy=(-0.20, H - PREFIX + 0.35), xytext=(-0.20, H + 0.30),
                        arrowprops=dict(arrowstyle="-", color=MUTED, lw=1.1))
            ax.text(-0.28, H - PREFIX / 2 + 0.4, "search\nprefix", ha="right",
                    va="center", fontsize=7.8, color=INK2)
            ax.annotate("", xy=(-0.20, -0.05), xytext=(-0.20, H - PREFIX + 0.15),
                        arrowprops=dict(arrowstyle="-", color=MUTED, lw=1.1))
            ax.text(-0.28, (H - PREFIX) / 2 - 0.1, "generated\ncontinuation",
                    ha="right", va="center", fontsize=7.8, color=INK2)

        if pi == 0:
            y = H - PREFIX
            ax.plot([-0.04, 0.94], [y, y], color=RED, lw=2.6, zorder=6)
            ax.text(1.00, y, "seam", ha="left", va="center", fontsize=8.4,
                    color=RED, fontweight="bold", zorder=7)
        if pi == 1:
            ax.add_patch(Rectangle((-0.06, H - PREFIX + 0.02), 1.02, PREFIX - 0.06,
                                   fc="none", ec=RED, lw=1.8, ls=(0, (3, 2)), zorder=6))
            ax.text(1.00, H - PREFIX / 2 + 0.4, "clamped\nclean rows", ha="left",
                    va="center", fontsize=8.2, color=RED, fontweight="bold",
                    linespacing=1.4)
        if pi == 2:
            ax.add_patch(Rectangle((-0.06, -0.06), 1.02, H + 0.06, fc="none",
                                   ec=GOOD, lw=1.6, zorder=6))

        ax.text(0.45, -0.95, how, ha="center", va="top", fontsize=7.9, color=INK2,
                linespacing=1.55)
        ax.text(0.45, -3.55, note, ha="center", va="top", fontsize=7.9,
                color=flag, fontweight="bold", linespacing=1.55)

    # shared sequential legend for the noise encoding
    cax = fig.add_axes([0.32, -0.045, 0.36, 0.020])
    grad = np.linspace(0, 1, 256).reshape(1, -1)
    cax.imshow(grad, aspect="auto", cmap=NOISE_CMAP)
    cax.set_xticks([]); cax.set_yticks([])
    for s in cax.spines.values():
        s.set_edgecolor(BASE)
    cax.text(-0.02, 0.5, "clean (k = 0)", transform=cax.transAxes, ha="right",
             va="center", fontsize=7.6, color=INK2)
    cax.text(1.02, 0.5, "pure noise (k = K)", transform=cax.transAxes, ha="left",
             va="center", fontsize=7.6, color=INK2)

    fig.suptitle("Three ways to condition a continuation on a search prefix",
                 fontsize=10.8, fontweight="bold", color=INK, y=1.01)
    save(fig, "fig3_2_expansion_mechanisms")


# ════════════════════════════════════════════════════════════════════════
# Figure 3-3 — the Diffusion-Forcing scheduling matrix
# ════════════════════════════════════════════════════════════════════════
def fig_schedule():
    T, K, HIST, slope = 32, 20, 4, 1
    sweeps = np.arange(K + T - 1, -1, -1)          # execution order: noisy -> clean
    M = np.zeros((len(sweeps), T))
    for i, m in enumerate(sweeps):
        M[i] = np.clip(m - slope * (T - 1 - np.arange(T)), 0, K)
    M[:, :HIST] = 0                                # history forced to level 0

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    im = ax.imshow(M, aspect="auto", cmap=NOISE_CMAP, vmin=0, vmax=K,
                   interpolation="nearest")
    ax.axvline(HIST - 0.5, color=RED, lw=2.0, zorder=5)
    ax.text(HIST + 0.6, len(sweeps) * 0.90, "clean history\n(level 0, forced)",
            ha="left", va="center", fontsize=8.4, color=RED, fontweight="bold",
            zorder=7)

    ax.set_xlabel("token position along the plan window  (t)")
    ax.set_ylabel("denoising sweep  (execution order, top to bottom)")
    ax.set_title("The scheduling matrix: the near future resolves before the far future")
    ax.set_yticks([0, len(sweeps) // 2, len(sweeps) - 1])
    ax.set_yticklabels(["first", "...", "last"])
    for s in ax.spines.values():
        s.set_edgecolor(BASE)

    cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.045)
    cb.set_label("per-token noise level  k", fontsize=8.6, color=INK2)
    cb.outline.set_edgecolor(BASE)

    ax.annotate("far future still\nnear pure noise", xy=(T - 2, len(sweeps) * 0.42),
                xytext=(T - 12.5, len(sweeps) * 0.20), fontsize=8.0, color=INK,
                arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.1))
    fig.text(0.5, -0.035,
             f"T = {T} tokens, K = {K} levels, slope = {slope}  =>  {len(sweeps)} sweeps. "
             "Each row is one parallel forward pass over all tokens.",
             ha="center", fontsize=7.9, color=MUTED)
    save(fig, "fig3_3_scheduling_matrix")


# ════════════════════════════════════════════════════════════════════════
# Figure 3-4 — composed-window scoring
# ════════════════════════════════════════════════════════════════════════
def fig_composed():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.8, 3.6),
                                   gridspec_kw={"width_ratios": [1, 1.35]})
    blank(axL); blank(axR)

    # -- left: the tree --------------------------------------------------
    axL.text(0.5, 0.96, "The search tree", ha="center", va="top", fontsize=9.8,
             color=INK, fontweight="bold")
    nodes = {"root": (0.5, 0.76), "a": (0.26, 0.50), "b": (0.74, 0.50),
             "a1": (0.13, 0.22), "a2": (0.39, 0.22), "b1": (0.87, 0.22)}
    for k, (x, y) in nodes.items():
        hl = k in ("root", "a", "a2")
        axL.add_patch(plt.Circle((x, y), 0.055, fc=AQUA if hl else "#eef1ee",
                                 ec=AQUA if hl else BASE, lw=1.5, zorder=4))
    for p, c in [("root", "a"), ("root", "b"), ("a", "a1"), ("a", "a2"), ("b", "b1")]:
        hl = (p, c) in [("root", "a"), ("a", "a2")]
        axL.plot(*zip(nodes[p], nodes[c]), color=AQUA if hl else BASE,
                 lw=2.2 if hl else 1.2, zorder=2)
    axL.text(nodes["root"][0], nodes["root"][1] + 0.10, "$s_t$  (root)", ha="center",
             fontsize=8.4, color=INK2)
    axL.text(0.39, 0.10, "depth 2 node", ha="center", fontsize=8.2, color=AQUA,
             fontweight="bold")
    axL.text(0.5, 0.005, "each edge commits L waypoints", ha="center", fontsize=7.8,
             color=MUTED, style="italic")

    # -- right: window alignment ----------------------------------------
    axR.text(0.5, 0.96, "Every node is scored on the SAME window", ha="center",
             va="top", fontsize=9.8, color=INK, fontweight="bold")
    rows = [("root", 0), ("depth 1", 1), ("depth 2", 2)]
    x0, wtot, ncell = 0.20, 0.66, 12
    cw = wtot / ncell
    for i, (lab, d) in enumerate(rows):
        y = 0.66 - i * 0.155
        for c in range(ncell):
            fc = AQUA if c < d else NOISE_CMAP(0.18)
            axR.add_patch(Rectangle((x0 + c * cw, y), cw * 0.9, 0.085, fc=fc,
                                    ec="white", lw=0.6, zorder=3))
        axR.text(x0 - 0.02, y + 0.042, lab, ha="right", va="center", fontsize=8.2,
                 color=INK2)
    axR.plot([x0, x0], [0.335, 0.79], color=INK2, lw=1.0, ls=(0, (2, 2)))
    axR.plot([x0 + wtot, x0 + wtot], [0.335, 0.79], color=INK2, lw=1.0, ls=(0, (2, 2)))
    axR.text(x0 + wtot / 2, 0.83, "shared window  [ t ,  t + H )", ha="center",
             fontsize=8.4, color=INK, fontweight="bold")

    axR.annotate("committed prefix", xy=(x0 + cw * 0.55, 0.345),
                 xytext=(x0 + wtot * 0.34, 0.272), fontsize=7.8, color=AQUA,
                 fontweight="bold", ha="left", va="center",
                 arrowprops=dict(arrowstyle="-|>", color=AQUA, lw=1.0))

    axR.text(x0 + wtot / 2, 0.185, "naive alternative", ha="center", va="center",
             fontsize=7.8, color=CRIT, fontweight="bold")
    axR.text(x0 + wtot / 2, 0.088,
             "scoring each node on its own shifted window is biased: later windows\n"
             "hold more near-goal steps, so a maximising backup would reward\n"
             "visiting rather than choosing well",
             ha="center", va="center", fontsize=7.5, color=MUTED, linespacing=1.5)
    save(fig, "fig3_4_composed_window")


# ════════════════════════════════════════════════════════════════════════
# Figure 3-5 — the per-token guidance shift and its self-annealing
# ════════════════════════════════════════════════════════════════════════
def fig_guidance():
    T, K, HIST = 24, 20, 6
    abar = cosine_abar(K)
    kvec = np.clip(np.round(np.linspace(-6, K, T)), 0, K).astype(int)
    kvec[:HIST] = 0                                        # clean history
    weight = np.sqrt(1 - abar[kvec])                       # per-token guidance scale

    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    ax.yaxis.grid(True, color=GRID, lw=0.6, zorder=0); ax.set_axisbelow(True)

    for t in range(T):
        ax.add_patch(Rectangle((t - 0.42, -0.13), 0.84, 0.10,
                               fc=NOISE_CMAP(kvec[t] / K), ec="white", lw=0.5,
                               zorder=3, clip_on=False))
    ax.bar(np.arange(T), weight, width=0.62, color=BLUE, zorder=3,
           edgecolor="white", linewidth=0.7)

    ax.axvspan(-0.6, HIST - 0.5, color="#f3f3f1", zorder=1)
    ax.text((HIST - 1) / 2, 0.42, "clean history\n(no push)", ha="center", va="center",
            fontsize=8.2, color=INK2)
    ax.annotate("far future:\nfull guidance weight", xy=(T - 2.4, weight[-3]),
                xytext=(T - 11.5, 0.52), fontsize=8.4, color=INK, ha="center",
                arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.1))

    ax.set_xlabel("token position along the plan window  (t)")
    ax.set_ylabel(r"guidance scale   $\sqrt{1-\bar{\alpha}[k_t]}$")
    ax.set_title("Guidance self-anneals independently at every token")
    ax.set_xlim(-0.7, T - 0.3); ax.set_ylim(-0.13, 1.26)
    ax.set_xticks([0, 6, 12, 18, 23])
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    ax.text(0.5, 1.20,
            r"$\varepsilon \leftarrow \varepsilon - w\,\sqrt{1-\bar{\alpha}[k]}\;"
            r"\nabla_x V(x,\,k)$",
            transform=ax.transData, ha="center", va="center", fontsize=11.5,
            color=INK)
    fig.text(0.5, -0.03,
             "A value model conditioned on a single trajectory-level noise scalar cannot express this profile: "
             "it has one weight for the whole window.",
             ha="center", fontsize=7.9, color=MUTED)
    save(fig, "fig3_5_guidance_shift")


# ════════════════════════════════════════════════════════════════════════
# Figure 2-1 — the forward and reverse diffusion processes, on a trajectory
# ════════════════════════════════════════════════════════════════════════
def fig_diffusion():
    """What noising actually does to a PLAN, not to an image: a smooth path
    degraded to noise and back. Aimed at a reader outside the area."""
    rng = np.random.default_rng(0)
    T = 60
    s = np.linspace(0, 1, T)
    path = np.stack([s * 4.0 + 0.6 * np.sin(3.4 * s),
                     1.6 * np.sin(2.1 * s) + 0.35 * s], 1)          # a plausible plan
    K = 20
    abar = cosine_abar(K)
    levels = [0, 4, 9, 14, 20]
    eps = rng.standard_normal(path.shape)

    fig, axes = plt.subplots(1, len(levels), figsize=(9.6, 2.9))
    for ax, k in zip(axes, levels):
        x = np.sqrt(abar[k]) * path + np.sqrt(1 - abar[k]) * eps * 1.1
        ax.plot(x[:, 0], x[:, 1], color=NOISE_CMAP(0.15 + 0.8 * k / K), lw=1.9,
                solid_capstyle="round", zorder=3)
        ax.scatter([x[0, 0]], [x[0, 1]], s=22, color=AQUA, zorder=4,
                   edgecolor="white", linewidth=0.8)
        ax.set_xlim(-2.4, 6.2); ax.set_ylim(-3.2, 3.4)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(BASE)
        ax.set_title(f"k = {k}" + ("   (data)" if k == 0 else "   (noise)" if k == K else ""),
                     fontsize=9, color=INK, pad=6)
        ax.text(0.5, -0.10, r"$\bar{\alpha}_k$ = " + f"{abar[k]:.2f}",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=8, color=MUTED)

    fig.subplots_adjust(top=0.70, bottom=0.20, wspace=0.16)
    # forward / reverse arrows spanning the row
    fig.patches.append(FancyArrowPatch((0.09, 0.90), (0.93, 0.90),
        transform=fig.transFigure, arrowstyle="-|>", mutation_scale=14,
        lw=1.5, color=INK2, zorder=5))
    fig.text(0.51, 0.925,
             r"forward process  $q(x_k \mid x_{k-1})$  —  add Gaussian noise, fixed",
             ha="center", fontsize=8.6, color=INK2)
    fig.patches.append(FancyArrowPatch((0.93, 0.055), (0.09, 0.055),
        transform=fig.transFigure, arrowstyle="-|>", mutation_scale=14,
        lw=1.5, color=BLUE, zorder=5))
    fig.text(0.51, 0.005,
             r"reverse process  $p_\theta(x_{k-1} \mid x_k)$  —  predict the noise "
             "and remove it, learned",
             ha="center", fontsize=8.6, color=BLUE)
    save(fig, "fig2_1_diffusion_process")


if __name__ == "__main__":
    print("generating concept figures -> figures/")
    fig_diffusion()
    fig_system()
    fig_expansion()
    fig_schedule()
    fig_composed()
    fig_guidance()
    print("done.")
