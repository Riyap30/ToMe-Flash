from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def draw_flowchart(title, steps, notes, colors, output_path):
    fig, ax = plt.subplots(figsize=(10, 14), dpi=200)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.patch.set_facecolor("white")
    ax.text(
        0.5,
        0.965,
        title,
        ha="center",
        va="top",
        fontsize=20,
        fontweight="bold",
        color=colors["title"],
    )

    x = 0.08
    w = 0.66
    h = 0.075
    top = 0.90
    gap = 0.02

    ys = []
    for i, step in enumerate(steps):
        y = top - (h + gap) * i - h
        ys.append(y)
        box = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            linewidth=2,
            edgecolor=colors["edge"],
            facecolor=colors["fill1"] if i % 2 == 0 else colors["fill2"],
        )
        ax.add_patch(box)
        ax.text(
            x + 0.02,
            y + h / 2,
            step,
            va="center",
            ha="left",
            fontsize=11.5,
            color=colors["text"],
        )

    for i in range(len(steps) - 1):
        y1 = ys[i]
        y2 = ys[i + 1] + h
        arrow = FancyArrowPatch(
            (x + w / 2, y1),
            (x + w / 2, y2),
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=2,
            color=colors["arrow"],
        )
        ax.add_patch(arrow)

    note_x = 0.78
    note_w = 0.19
    note_h = 0.15
    note_start_y = 0.67
    note_gap = 0.03
    for i, note in enumerate(notes):
        ny = note_start_y - i * (note_h + note_gap)
        nbox = FancyBboxPatch(
            (note_x, ny),
            note_w,
            note_h,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            linewidth=1.8,
            edgecolor=colors["note_edge"],
            facecolor=colors["note_fill"],
        )
        ax.add_patch(nbox)
        ax.text(
            note_x + 0.01,
            ny + note_h / 2,
            note,
            va="center",
            ha="left",
            fontsize=10.5,
            color=colors["text"],
            wrap=True,
        )

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main():
    figures_dir = Path("final results/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    tome_steps = [
        "1) Load pretrained DeiT-B/16 via timm.create_model()",
        "2) Set eval mode, move to device, cast model to bfloat16",
        "3) Patch fresh model in-place with tome.patch.timm(model)",
        "4) Set merge ratio r using model.r = r",
        "5) Per each of 12 layers, bipartite soft matching merges r token pairs",
        "6) Token sequence shrinks progressively: 197 -> (197 - 12r)",
        "7) Forward pass with reduced token set",
        "8) Classification logits output (1000 classes)",
    ]
    tome_notes = [
        "Constraint:\nPatch only a fresh model.\nDo not re-patch an already patched model.",
        "Sweep used in repo:\nr in {4, 8, 12, 14, 16}",
    ]
    tome_colors = {
        "title": "#0B3D91",
        "edge": "#0B3D91",
        "fill1": "#EAF2FF",
        "fill2": "#D8F3F0",
        "text": "#111827",
        "arrow": "#0F766E",
        "note_fill": "#FFF7E6",
        "note_edge": "#B45309",
    }

    flash_steps = [
        "1) Load pretrained DeiT-B/16 via timm.create_model()",
        "2) Set eval mode, move to device, cast model to bfloat16",
        "3) Replace each block.attn with FlashAttentionBlock",
        "4) Reuse original qkv/proj weights (no reinitialization)",
        "5) Compute qkv then reshape to (B, N, 3, H, D) and split q, k, v",
        "6) Run flash_attn_func(q, k, v, dropout_p=0.0, causal=False)",
        "7) Reshape output to (B, N, C) and apply output projection",
        "8) Sanity check with dummy input: output shape must be (1, 1000)",
        "9) Use for inference, benchmarking, and accuracy evaluation",
    ]
    flash_notes = [
        "Only attention kernel is replaced.",
        "MLP, norms, and classifier head remain unchanged.",
    ]
    flash_colors = {
        "title": "#1E3A8A",
        "edge": "#1E3A8A",
        "fill1": "#EEF2FF",
        "fill2": "#FFEFD5",
        "text": "#111827",
        "arrow": "#C2410C",
        "note_fill": "#ECFEFF",
        "note_edge": "#0E7490",
    }

    draw_flowchart(
        "Token Merging (ToMe) Methodology - DeiT-B/16",
        tome_steps,
        tome_notes,
        tome_colors,
        figures_dir / "tome_methodology_flowchart.png",
    )

    draw_flowchart(
        "FlashAttention-2 Methodology - DeiT-B/16",
        flash_steps,
        flash_notes,
        flash_colors,
        figures_dir / "flash_attention_methodology_flowchart.png",
    )


if __name__ == "__main__":
    main()
