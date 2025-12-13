#!/usr/bin/env python3
"""
yolo_plots_from_results.py

Reads a YOLO training `results.csv` (same format as YOLOv5/YOLOv8) and:
 - cleans the CSV
 - generates individual plots (one metric per PNG) using matplotlib
 - combines all generated PNGs into a single tiled PNG
 - writes outputs to ./yolo_results_plots (or change OUT_DIR)
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import math
import os

# --- Config ---
DATA_PATH = Path("results.csv")   # change if your CSV is elsewhere
OUT_DIR = Path("yolo_results_plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Create sample results.csv if not provided (for quick testing) ---
if not DATA_PATH.exists():
    example = """epoch,time,train/box_loss,train/cls_loss,train/dfl_loss,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B),val/box_loss,val/cls_loss,val/dfl_loss,lr/pg0,lr/pg1,lr/pg2
1,140.093,4.05793,4.68332,3.62277,0.37866,0.16917,0.07857,0.02384,3.27511,2.99234,3.18228,0.0820431,0.00199522,0.00199522
2,268.512,2.92993,2.80218,2.65857,0.2826,0.43089,0.26927,0.11188,2.32951,2.02849,2.14399,0.0640352,0.0039873,0.0039873
3,400.121,2.10012,2.20011,1.90005,0.45012,0.52033,0.39021,0.15011,1.83001,1.64022,1.75033,0.0450000,0.0050000,0.0050000
"""
    DATA_PATH.write_text(example)

# --- Read CSV robustly ---
df = pd.read_csv(DATA_PATH)
df = df.dropna(how="all")

# normalize column names to python-friendly
clean_cols = {}
for c in df.columns:
    new = c.strip().replace("/", "_").replace("(", "").replace(")", "").replace("-", "_").replace(" ", "_")
    new = new.replace("metrics_", "")
    clean_cols[c] = new
df = df.rename(columns=clean_cols)

if "epoch" not in df.columns:
    df.insert(0, "epoch", range(1, len(df) + 1))

for col in df.columns:
    try:
        df[col] = pd.to_numeric(df[col])
    except Exception:
        pass

# Save processed
processed_csv = OUT_DIR / "processed_results.csv"
df.to_csv(processed_csv, index=False)

# plotting helper
def save_line_plot(x, y, xlabel, ylabel, title, outpath, markers=True):
    plt.figure(figsize=(8,5))
    if markers:
        plt.plot(x, y, marker='o')
    else:
        plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle='--', linewidth=0.4, alpha=0.7)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

plots = []
# common loss columns
loss_names = ["box_loss", "cls_loss", "dfl_loss"]
for name in loss_names:
    train_col = f"train_{name}"
    val_col = f"val_{name}"
    if train_col in df.columns:
        plots.append((train_col, "loss", f"Train {name.replace('_',' ').title()}", f"train_{name}.png"))
    if val_col in df.columns:
        plots.append((val_col, "loss", f"Validation {name.replace('_',' ').title()}", f"val_{name}.png"))

# metrics detection
for col in df.columns:
    lname = col.lower()
    if "precision" in lname and col not in [p[0] for p in plots]:
        plots.append((col, "value", f"Precision ({col})", f"{col}.png"))
    if "recall" in lname and col not in [p[0] for p in plots]:
        plots.append((col, "value", f"Recall ({col})", f"{col}.png"))
    if "map50_95" in lname and col not in [p[0] for p in plots]:
        plots.append((col, "mAP", f"mAP50-95 ({col})", f"{col}.png"))
    if "map50" in lname and "map50_95" not in lname and col not in [p[0] for p in plots]:
        plots.append((col, "mAP", f"mAP50 ({col})", f"{col}.png"))

# learning rate style columns (various possible names)
for col in df.columns:
    if col.startswith("pg") or col.startswith("lr_") or col.startswith("lrpg") or col.startswith("lr_pg"):
        plots.append((col, "lr", f"Learning Rate ({col})", f"{col}.png"))

# ensure some totals
train_loss_cols = [c for c in df.columns if c.startswith("train_") and c.endswith("loss")]
if train_loss_cols:
    df["train_total_loss"] = df[train_loss_cols].sum(axis=1)
    plots.append(("train_total_loss", "loss", "Train Total Loss", "train_total_loss.png"))

val_loss_cols = [c for c in df.columns if c.startswith("val_") and c.endswith("loss")]
if val_loss_cols:
    df["val_total_loss"] = df[val_loss_cols].sum(axis=1)
    plots.append(("val_total_loss", "loss", "Validation Total Loss", "val_total_loss.png"))

x = df["epoch"] if "epoch" in df.columns else df.index + 1

saved_files = []
for series, ylabel, title, fname in plots:
    try:
        y = df[series]
        outpath = OUT_DIR / fname
        save_line_plot(x, y, "epoch", ylabel, title, outpath)
        saved_files.append(outpath)
    except Exception:
        pass

# combine into a single tiled image (without matplotlib subplots)
from PIL import Image
images = [Image.open(str(p)) for p in saved_files]
if images:
    widths, heights = zip(*(i.size for i in images))
    target_width = max(widths)
    resized = []
    for im in images:
        w,h = im.size
        new_h = int(h * (target_width / w))
        resized.append(im.resize((target_width, new_h)))
    n = len(resized)
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    max_h_per_row = [0]*rows
    for idx,im in enumerate(resized):
        r = idx // cols
        max_h_per_row[r] = max(max_h_per_row[r], im.size[1])
    total_height = sum(max_h_per_row)
    total_width = cols * target_width
    combined = Image.new("RGB", (total_width, total_height), color=(255,255,255))
    y_offset = 0
    idx = 0
    for r in range(rows):
        x_offset = 0
        for c in range(cols):
            if idx >= n: break
            im = resized[idx]
            combined.paste(im, (x_offset, y_offset))
            x_offset += target_width
            idx += 1
        y_offset += max_h_per_row[r]
    combined_path = OUT_DIR / "results_combined.png"
    combined.save(combined_path, dpi=(150,150))

print("Saved processed CSV:", processed_csv)
print("Saved files in:", OUT_DIR)
