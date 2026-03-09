"""Generate progress.png from results.tsv."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import csv

# Read results
experiments = []
with open("results.tsv") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for i, row in enumerate(reader):
        bpb = float(row["val_bpb"])
        status = row["status"]
        desc = row["description"]
        if bpb == 0.0:  # crashes
            continue
        experiments.append((i, bpb, status, desc))

indices = [e[0] for e in experiments]
bpbs = [e[1] for e in experiments]
statuses = [e[2] for e in experiments]
descs = [e[3] for e in experiments]

# Track running best
running_best = []
best_so_far = float("inf")
for bpb in bpbs:
    if bpb < best_so_far:
        best_so_far = bpb
    running_best.append(best_so_far)

# Colors
colors = []
for s in statuses:
    if s == "keep":
        colors.append("#2ecc71")  # green
    else:
        colors.append("#e74c3c")  # red

fig, ax = plt.subplots(figsize=(16, 7))

# Plot all experiments as scatter
ax.scatter(range(len(bpbs)), bpbs, c=colors, s=50, zorder=3, edgecolors="white", linewidths=0.5)

# Plot running best line
ax.step(range(len(bpbs)), running_best, where="post", color="#2ecc71", linewidth=2.5, alpha=0.8, label="Running best")

# Annotate the kept experiments
kept = [(i, bpb, desc) for i, (_, bpb, status, desc) in enumerate(experiments) if status == "keep"]
for i, bpb, desc in kept:
    ax.annotate(
        desc,
        (i, bpb),
        textcoords="offset points",
        xytext=(10, -15 if i > 0 else 10),
        fontsize=8,
        fontweight="bold",
        color="#2ecc71",
        arrowprops=dict(arrowstyle="-", color="#2ecc71", lw=0.8) if i > 0 else None,
    )

# Annotate the worst outliers
for i, bpb, desc in [(idx, b, d) for idx, (_, b, s, d) in enumerate(experiments) if b > 1.5]:
    ax.annotate(
        f"{desc}\n(val_bpb={bpb:.2f})",
        (i, bpb),
        textcoords="offset points",
        xytext=(10, -10),
        fontsize=7,
        color="#e74c3c",
        alpha=0.7,
    )

# Styling
ax.set_xlabel("Experiment #", fontsize=12)
ax.set_ylabel("val_bpb (lower is better)", fontsize=12)
ax.set_title("autoresearch/mar9 — 66 Experiments, 3 Improvements Kept", fontsize=14, fontweight="bold")
ax.axhline(y=0.967531, color="gray", linestyle="--", alpha=0.5, label="Baseline (0.9675)")
ax.axhline(y=0.902167, color="#2ecc71", linestyle="--", alpha=0.5, label="Best (0.9022)")

# Filter y-axis to useful range (exclude extreme outliers for readability)
ax.set_ylim(0.88, 1.05)

# Legend
keep_patch = mpatches.Patch(color="#2ecc71", label="Kept")
discard_patch = mpatches.Patch(color="#e74c3c", label="Discarded")
ax.legend(handles=[keep_patch, discard_patch,
                    plt.Line2D([0], [0], color="#2ecc71", linewidth=2.5, label="Running best"),
                    plt.Line2D([0], [0], color="gray", linestyle="--", label="Baseline")],
          loc="upper right", fontsize=9)

ax.grid(True, alpha=0.2)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("progress.png", dpi=150, bbox_inches="tight")
print("Saved progress.png")
