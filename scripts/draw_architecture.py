"""Render the GroundedDNA v29 architecture (forward + losses) as a PNG.

Layout:
- Top strip (y > 6.5): forward pipeline left -> right
    inputs -> frozen encoders -> trainable adapters -> Sinkhorn-OT router
    -> 6-slot codebook quantizer -> codon heads -> DNA code
- Middle strip (y ~ 4.5-6.0): view-2 mini-pipeline + intermediate tensors
- Bottom strip (y < 4.0): all loss boxes (red), each fed by dashed arrows
                          from the relevant tensors above

Solid black arrows = data flow.  Dashed red arrows = loss target.
"""
from __future__ import annotations
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle


# ---------------------- helpers ---------------------- #

def box(ax, x, y, w, h, text, fc, ec="black", fontsize=8.5, weight="normal"):
    p = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.05",
        linewidth=1.0, facecolor=fc, edgecolor=ec,
    )
    ax.add_patch(p)
    ax.text(
        x + w / 2, y + h / 2, text, ha="center", va="center",
        fontsize=fontsize, fontweight=weight,
    )


def arrow(ax, p0, p1, color="black", lw=1.2, mutation_scale=12):
    a = FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=mutation_scale,
        linewidth=lw, color=color, shrinkA=2, shrinkB=2,
    )
    ax.add_patch(a)


def loss_arrow(ax, p0, p1, color="firebrick", lw=1.0):
    a = FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=10,
        linewidth=lw, color=color, linestyle=(0, (4, 3)),
        shrinkA=2, shrinkB=2,
    )
    ax.add_patch(a)


# ---------------------- color palette ---------------------- #

C_INPUT  = "#F0EAD6"   # cream
C_FROZEN = "#D9D9D9"   # gray (frozen encoders)
C_TRAIN  = "#BBDEFB"   # blue (trainable adapters)
C_ROUTE  = "#FFE0B2"   # orange (router internals)
C_CB     = "#E1BEE7"   # purple (codebooks / quantization)
C_CODON  = "#C8E6C9"   # green (codon heads / DNA code)
C_LOSS   = "#FFCDD2"   # pinkish red (loss boxes)


# ---------------------- canvas ---------------------- #

W, H = 26.0, 16.0
fig, ax = plt.subplots(figsize=(W, H), dpi=130)
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")

ax.text(
    W / 2, H - 0.45,
    "GroundedDNA  (v29: paired-aug NtXent on DNA codes)",
    ha="center", va="center", fontsize=15, fontweight="bold",
)
ax.text(
    W / 2, H - 0.85,
    "Solid arrows = data flow.  Dashed (red) arrows = loss targets.  "
    "Frozen = SigLIP2 backbone (gray).",
    ha="center", va="center", fontsize=10, color="dimgray",
)


# ============================================================
# Legend (top-left)
# ============================================================

ax.add_patch(Rectangle((0.4, 13.3), 7.5, 1.0, facecolor="white",
                        edgecolor="black", linewidth=0.6))
ax.text(0.6, 14.10, "Legend", fontsize=9, fontweight="bold")
xleg = 0.6
for label, c in [
    ("frozen encoder", C_FROZEN),
    ("trainable adapter", C_TRAIN),
    ("OT routing", C_ROUTE),
    ("codebook / VQ", C_CB),
    ("codon / DNA", C_CODON),
    ("loss term", C_LOSS),
]:
    ax.add_patch(Rectangle((xleg, 13.50), 0.30, 0.22,
                            facecolor=c, edgecolor="black", linewidth=0.5))
    ax.text(xleg + 0.36, 13.62, label, fontsize=8, va="center")
    xleg += 1.20


# ============================================================
# 1. INPUTS  (column 1, x = 0.3 .. 2.2)
# ============================================================

box(ax, 0.3, 11.9, 2.0, 0.9,
    "Image  $x$\n[B, 3, 224, 224]", C_INPUT, fontsize=9.5)
box(ax, 0.3, 10.5, 2.0, 0.9,
    "Image  $x'$  (paired aug)\n[B, 3, 224, 224]", C_INPUT, fontsize=9.5)
box(ax, 0.3, 8.6, 2.0, 1.1,
    "Qwen2.5-VL\n6 part captions\n[B, 6, L]", C_INPUT, fontsize=9.5)


# ============================================================
# 2. FROZEN SigLIP2 ENCODERS  (column 2, x = 2.7 .. 5.0)
# ============================================================

box(ax, 2.7, 11.9, 2.3, 0.9,
    "SigLIP2 Vision\n(frozen)", C_FROZEN, fontsize=9.5, weight="bold")
box(ax, 2.7, 10.5, 2.3, 0.9,
    "SigLIP2 Vision\n(view 2, frozen)", C_FROZEN, fontsize=9.5, weight="bold")
box(ax, 2.7, 8.6, 2.3, 1.1,
    "SigLIP2 Text\n(frozen)", C_FROZEN, fontsize=9.5, weight="bold")

arrow(ax, (2.3, 12.35), (2.7, 12.35))
arrow(ax, (2.3, 10.95), (2.7, 10.95))
arrow(ax, (2.3, 9.15),  (2.7, 9.15))


# ============================================================
# 3. RAW FEATURES  (column 3, x = 5.4 .. 7.6)
# ============================================================

box(ax, 5.4, 12.45, 2.2, 0.55,
    r"$visual\_tokens$  [B, 196, $H_v$]", C_FROZEN, fontsize=8.5)
box(ax, 5.4, 11.80, 2.2, 0.55,
    r"$visual\_global$  [B, $D_p$]", C_FROZEN, fontsize=8.5)
box(ax, 5.4, 11.05, 2.2, 0.55,
    r"$visual\_tokens'$  [B, 196, $H_v$]", C_FROZEN, fontsize=8.5)
box(ax, 5.4, 10.40, 2.2, 0.55,
    r"$visual\_global'$  [B, $D_p$]", C_FROZEN, fontsize=8.5)
box(ax, 5.4, 8.6,  2.2, 1.1,
    r"$text\_part\_raw$" + "\n[B, 6, $D_p$]", C_FROZEN, fontsize=8.5)

arrow(ax, (5.0, 12.35), (5.4, 12.70))
arrow(ax, (5.0, 12.35), (5.4, 12.05))
arrow(ax, (5.0, 10.95), (5.4, 11.30))
arrow(ax, (5.0, 10.95), (5.4, 10.65))
arrow(ax, (5.0, 9.15),  (5.4, 9.15))


# ============================================================
# 4. TRAINABLE ADAPTERS  (column 4, x = 8.0 .. 9.9)
# ============================================================

box(ax, 8.0, 12.45, 1.9, 0.55, "Visual Adapter (MLP)",       C_TRAIN, fontsize=8.5)
box(ax, 8.0, 11.80, 1.9, 0.55, "Global Adapter (Linear)",    C_TRAIN, fontsize=8.5)
box(ax, 8.0, 11.05, 1.9, 0.55, "Visual Adapter (shared)",    C_TRAIN, fontsize=8.5)
box(ax, 8.0, 10.40, 1.9, 0.55, "Global Adapter (shared)",    C_TRAIN, fontsize=8.5)
box(ax, 8.0, 8.6,   1.9, 1.1,  "Text Adapter\n(MLP)\n[B, 6, D]", C_TRAIN, fontsize=8.5)

arrow(ax, (7.6, 12.70), (8.0, 12.70))
arrow(ax, (7.6, 12.05), (8.0, 12.05))
arrow(ax, (7.6, 11.30), (8.0, 11.30))
arrow(ax, (7.6, 10.65), (8.0, 10.65))
arrow(ax, (7.6, 9.15),  (8.0, 9.15))


# ============================================================
# 5. SINKHORN-OT ROUTER (DETAILED)   (column 5, x = 10.4 .. 14.4)
# ============================================================

ax.add_patch(Rectangle(
    (10.4, 7.3), 4.0, 5.50, facecolor="#FFF6E6",
    edgecolor="#C9A36B", linewidth=1.2, linestyle="--",
))
ax.text(12.4, 12.65, "Sinkhorn-OT Router  (5 local parts)",
        ha="center", va="bottom", fontsize=10, fontweight="bold",
        color="#8B5A00")

# step1
box(ax, 10.6, 11.90, 3.6, 0.55,
    r"$\mathrm{sim}=\hat V \, \hat T^\top$  $\in[-1,1]^{B\times N\times 5}$",
    C_ROUTE, fontsize=8.5)
# step2
box(ax, 10.6, 11.10, 3.6, 0.55,
    r"$\mathrm{cost}=1-\mathrm{sim}$  $\Rightarrow$  $\log K=-\mathrm{cost}/\varepsilon$",
    C_ROUTE, fontsize=8.5)
# step3
box(ax, 10.6, 10.30, 3.6, 0.55,
    r"uniform marginals  $a=1/N,\ b=1/M$",
    C_ROUTE, fontsize=8.5)
# step4
box(ax, 10.6, 9.50, 3.6, 0.55,
    r"log-Sinkhorn  $\times$ 20  $\Rightarrow$  $P$  [B, N, 5]",
    C_ROUTE, fontsize=8.5)
# step5
box(ax, 10.6, 8.70, 3.6, 0.55,
    r"$z^{loc}_m = \sum_n P_{nm}V_n / \sum_n P_{nm}$",
    C_ROUTE, fontsize=8.5)
# step5b
box(ax, 10.6, 7.45, 3.6, 0.55,
    r"$ot\_cost = \langle P,\ \mathrm{cost}\rangle$  [B]",
    C_ROUTE, fontsize=8.5)

# vertical stack arrows inside router
for y0, y1 in [(11.90, 11.65), (11.10, 10.85), (10.30, 10.05),
               (9.50, 9.25), (8.70, 8.00)]:
    arrow(ax, (12.4, y0), (12.4, y1), lw=0.9, mutation_scale=8)

# router inputs
arrow(ax, (9.9, 12.70), (10.6, 12.30), color="dimgray")
arrow(ax, (9.9, 9.15),  (10.6, 12.10), color="dimgray")
ax.text(10.20, 11.90, "$V$", fontsize=9, color="dimgray")
ax.text(10.20, 10.20, "$T_{1..5}$", fontsize=8, color="dimgray")


# ============================================================
# 6. INTERMEDIATE TENSORS BEFORE QUANTIZATION  (column 6, x = 14.8 .. 16.7)
# ============================================================

box(ax, 14.8, 11.55, 1.9, 0.55, r"$z^{glob}$  [B, D]", C_TRAIN, fontsize=9)
box(ax, 14.8, 10.40, 1.9, 0.55,
    r"$z = [z^{glob}; z^{loc}]$" + "\n" + r"[B, 6, D]",
    "#F5E6F8", fontsize=8.5)
box(ax, 14.8, 8.85, 1.9, 0.55, r"$z^{loc}_{1..5}$  [B, 5, D]", C_ROUTE, fontsize=9)

# arrows
arrow(ax, (9.9, 11.80), (14.8, 11.80))                      # global_adapter -> z_global
arrow(ax, (14.2, 8.95), (14.8, 9.10))                       # router pool -> z_local
arrow(ax, (15.75, 11.55), (15.75, 10.95))                   # z_global -> concat
arrow(ax, (15.75, 8.85),  (15.75, 10.40))                   # z_local -> concat


# ============================================================
# 7. CODEBOOK QUANTIZER (DETAILED)  (column 7, x = 17.1 .. 22.0)
# ============================================================

ax.add_patch(Rectangle(
    (17.1, 7.6), 4.9, 5.20, facecolor="#F5E1F8",
    edgecolor="#7B1FA2", linewidth=1.2, linestyle="--",
))
ax.text(19.55, 12.65, "Per-slot VQ Codebook  (M=6,  K=64)",
        ha="center", va="bottom", fontsize=10, fontweight="bold",
        color="#4A148C")

# codebook tableau: 6 columns of 4 boxes each
ax.text(17.30, 12.20, r"codebooks $[6, K, D]$", fontsize=8, color="#4A148C")
for m in range(6):
    cx = 17.40 + m * 0.72
    for k in range(4):
        cy = 11.80 - k * 0.22
        ax.add_patch(Rectangle(
            (cx, cy), 0.55, 0.20, facecolor="#E1BEE7",
            edgecolor="#7B1FA2", linewidth=0.5,
        ))
    ax.text(cx + 0.275, 10.78, "$\\vdots$",
            fontsize=9, ha="center", color="#7B1FA2")
    ax.text(cx + 0.275, 11.04, "$k=K$",
            fontsize=6.5, ha="center", color="#4A148C")
    ax.text(cx + 0.275, 12.05, f"$m={m}$",
            fontsize=6.5, ha="center", color="#4A148C")

# argmin
box(ax, 17.20, 9.65, 4.7, 0.55,
    r"$k^*_m = \arg\min_k \|z_m - \mathrm{cb}[m,k]\|^2$",
    C_CB, fontsize=8.5)
# STE quantize
box(ax, 17.20, 8.85, 4.7, 0.55,
    r"$q_m = \mathrm{cb}[m, k^*_m]$  + STE",
    C_CB, fontsize=8.5)
# output q
box(ax, 17.20, 8.05, 4.7, 0.50,
    r"$q$  [B, 6, D]", "#D1C4E9", fontsize=9)

arrow(ax, (16.7, 10.55), (17.20, 9.95))                    # z -> argmin
arrow(ax, (19.55, 9.65), (19.55, 9.45), lw=0.9, mutation_scale=8)
arrow(ax, (19.55, 8.85), (19.55, 8.55), lw=0.9, mutation_scale=8)


# ============================================================
# 8. GLOBAL GATE  (between codebook and codon)
# ============================================================

box(ax, 17.10, 6.40, 4.9, 0.85,
    r"Global Gate  (off in v23b+):" + "\n"
    r"$\bar q_m = q^{loc}_m + \sigma(\alpha_m)\cdot q^{glob}$  for $m=1..5$",
    "#EDE7F6", fontsize=8.5)
arrow(ax, (19.55, 8.05), (19.55, 7.25))


# ============================================================
# 9. CODON HEADS + DNA CODE
# ============================================================

ax.add_patch(Rectangle(
    (17.10, 4.0), 4.9, 2.10, facecolor="#E8F5E9",
    edgecolor="#2E7D32", linewidth=1.2, linestyle="--",
))
ax.text(19.55, 6.0, "6 Codon Heads (one per codebook slot)",
        ha="center", va="bottom", fontsize=10, fontweight="bold",
        color="#1B5E20")

# 6 small boxes
for m in range(6):
    cx = 17.30 + m * 0.78
    ax.add_patch(Rectangle(
        (cx, 5.10), 0.66, 0.50, facecolor=C_CODON,
        edgecolor="#2E7D32", linewidth=0.7,
    ))
    ax.text(cx + 0.33, 5.35, f"$h_{m}$", ha="center", va="center",
            fontsize=8.5, fontweight="bold")
ax.text(19.55, 4.75,
        "each: split $D \\to 3 \\times (D/3)$, Linear$\\to$4, "
        "Gumbel-softmax (STE)",
        ha="center", fontsize=8, color="#1B5E20")

# DNA code output
box(ax, 17.10, 3.20, 4.9, 0.55,
    r"DNA code  $u^{ST} \in \{e_A,e_C,e_G,e_T\}^{18}$  [B, 18, 4]",
    "#A5D6A7", fontsize=9, weight="bold")

arrow(ax, (19.55, 6.40), (19.55, 5.60))                  # gate -> heads
arrow(ax, (19.55, 5.10), (19.55, 3.75))                  # heads -> DNA code


# ============================================================
# 10. VIEW-2 MINI PIPELINE (lower-middle)
# ============================================================

box(ax, 10.6, 5.20, 5.5, 0.55,
    "view 2: same Adapter $\\to$ Router $\\to$ Quantizer $\\to$ Codon",
    "#FFF3E0", fontsize=9)
box(ax, 10.6, 4.40, 5.5, 0.55,
    r"DNA code (view 2)  $u'^{ST}$  [B, 18, 4]",
    "#A5D6A7", fontsize=9)
arrow(ax, (5.4, 10.65), (10.6, 5.40), color="darkgray", lw=0.7,
      mutation_scale=8)
arrow(ax, (13.35, 5.20), (13.35, 4.95), lw=0.9, mutation_scale=8)


# ============================================================
# 11. LOSS LAYER (bottom strip)
# ============================================================

# Layout: y = 1.5 .. 2.7 (loss boxes) and y = 0.6 .. 1.4 (long-form)

# Top-row losses
box(ax, 18.5, 1.85, 4.0, 0.65,
    r"$L_\mathrm{ntxent}$  =  NtXent$_T$( $u^{ST}$, $u'^{ST}$ )  ★",
    C_LOSS, fontsize=10, weight="bold")
loss_arrow(ax, (19.55, 3.20), (20.5, 2.50))                  # u^ST  -> L_ntxent
loss_arrow(ax, (13.35, 4.40), (19.50, 2.50))                 # u'^ST -> L_ntxent

box(ax, 14.0, 1.85, 4.0, 0.65,
    r"$L_\mathrm{vq}=\|q-\mathrm{sg}(z)\|^2 + \beta\|z-\mathrm{sg}(q)\|^2$",
    C_LOSS, fontsize=8.5)
loss_arrow(ax, (15.75, 10.40), (16.0, 2.50))                 # z
loss_arrow(ax, (17.20, 8.30),  (16.5, 2.50))                 # q

box(ax, 9.5, 1.85, 4.0, 0.65,
    r"$L_\mathrm{quant}=\|u^c - \mathrm{sg}(u^h)\|^2$",
    C_LOSS, fontsize=8.5)
loss_arrow(ax, (19.55, 5.10),  (12.0, 2.50))                 # u^h via codon
loss_arrow(ax, (17.10, 3.45),  (11.5, 2.50))                 # u^c

box(ax, 5.0, 1.85, 4.0, 0.65,
    r"$L_\mathrm{anchor}=1-\mathrm{cos}(\bar{\mathrm{cb}}_{1..5},\ \mathrm{EMA}(T_{1..5}))$",
    C_LOSS, fontsize=8.5)
loss_arrow(ax, (17.40, 11.80), (8.5, 2.50))                  # codebook means
loss_arrow(ax, (9.9, 9.15),    (6.5, 2.50))                  # text anchor (T)

box(ax, 0.5, 1.85, 4.0, 0.65,
    r"$L_\mathrm{wass}=\mathbb{E}_b\,\langle P_b,\,\mathrm{cost}_b\rangle$",
    C_LOSS, fontsize=8.5)
loss_arrow(ax, (12.4, 7.45), (3.0, 2.50))                    # ot_cost

# Bottom-row losses
box(ax, 14.0, 1.0, 4.0, 0.65,
    r"$L_\mathrm{dna}=H(u^c) + \eta\cdot \|\bar u^c-\frac{1}{4}\|^2$",
    C_LOSS, fontsize=8.5)
loss_arrow(ax, (17.10, 3.45), (16.5, 1.65))                  # u^c

box(ax, 9.5, 1.0, 4.0, 0.65,
    r"$L_\mathrm{bu}=\|P_\mathrm{usage}-\frac{1}{K}\|^2 + \rho\|\,\mathrm{offdiag}\,G\|^2$",
    C_LOSS, fontsize=8.5)
loss_arrow(ax, (17.40, 9.65), (12.0, 1.65))                  # codebook distances

box(ax, 0.5, 1.0, 8.5, 0.65,
    "Disabled in v29:  "
    r"$L_\mathrm{hash}=\|\mathrm{sim}_\mathrm{DNA}-S\|^2$  "
    r"·  $L_\mathrm{hash\_hard}$  ·  "
    r"$L_\mathrm{recon}$  (decoder off)",
    "#F5F5F5", fontsize=8.5)


# ============================================================
# 12. v29 total
# ============================================================

box(ax, 0.5, 0.10, 25.0, 0.65,
    r"$L_\mathrm{total}^\mathrm{v29} = "
    r"\lambda_\mathrm{ntxent}\, L_\mathrm{ntxent}"
    r"+ \lambda_\mathrm{vq}\, L_\mathrm{vq}"
    r"+ \lambda_\mathrm{quant}\, L_\mathrm{quant}"
    r"+ \lambda_\mathrm{anchor}\, L_\mathrm{anchor}"
    r"+ \lambda_\mathrm{dna}\, L_\mathrm{dna}"
    r"+ \lambda_\mathrm{bu}\, L_\mathrm{bu}"
    r"+ \lambda_\mathrm{wass}\, L_\mathrm{wass}$"
    "    (★ primary retrieval signal in v29)",
    "#FFF8E1", fontsize=10, weight="bold")


# Save
out = "/home/yschoi/GroundedDNA/docs/architecture_v29.png"
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
print(f"saved -> {out}")
