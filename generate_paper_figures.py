import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


FEATURE_DIR = "feature_data"
RESULTS_DIR = "dl_comparison_results_gpu"
OUTPUT_DIR = os.path.join(RESULTS_DIR, "paper_figures")
ARCH_DIR = os.path.join(OUTPUT_DIR, "model_architectures")
DIM_REDUCTION_DIR = "dim_reduction_results"
os.makedirs(ARCH_DIR, exist_ok=True)


SINGLE_FEATURES = ["X_mel", "X_mfcc", "X_cqt", "X_stft"]
SINGLE_MODELS = ["SimpleCNN", "ResNet18", "ResNet34", "ResNet50", "RNN", "LSTM", "Transformer"]


def save_current_figure(save_stem):
    plt.savefig(save_stem + ".png", dpi=300, bbox_inches="tight")
    plt.savefig(save_stem + ".svg", dpi=300, bbox_inches="tight")
    plt.savefig(save_stem + ".pdf", dpi=300, bbox_inches="tight")
    plt.close()


def save_figure_with_paths(fig, save_stem):
    fig.savefig(save_stem + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(save_stem + ".svg", dpi=300, bbox_inches="tight")
    fig.savefig(save_stem + ".pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def copy_all_formats(src_stem, dst_stem):
    copied = []
    for ext in [".png", ".svg", ".pdf"]:
        src = src_stem + ext
        dst = dst_stem + ext
        if os.path.exists(src):
            with open(src, "rb") as fsrc:
                data = fsrc.read()
            with open(dst, "wb") as fdst:
                fdst.write(data)
            copied.append(dst)
    return copied


def create_image_grid(image_paths, titles, save_stem, ncols=2, figsize=(16, 12), suptitle=None):
    existing = [(img, title) for img, title in zip(image_paths, titles) if os.path.exists(img)]
    if not existing:
        return []

    n_images = len(existing)
    nrows = math.ceil(n_images / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]

    for ax, (img_path, title) in zip(axes, existing):
        ax.imshow(mpimg.imread(img_path))
        ax.set_title(title, fontsize=14)
        ax.axis("off")

    for idx in range(len(existing), len(axes)):
        axes[idx].axis("off")

    if suptitle:
        fig.suptitle(suptitle, fontsize=16, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.97])
    else:
        fig.tight_layout()

    save_figure_with_paths(fig, save_stem)
    return [save_stem + ext for ext in [".png", ".svg", ".pdf"]]


def add_box(ax, xy, width, height, text, facecolor="#E8F0FE", edgecolor="#1F4E79", fontsize=10):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.5,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(box)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", fontsize=fontsize)


def add_arrow(ax, start, end, color="#4A4A4A"):
    arrow = FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=12, linewidth=1.5, color=color)
    ax.add_patch(arrow)


def get_feature_shape(feature_name):
    if feature_name == "Fusion_Mel_MFCC":
        x1 = np.load(os.path.join(FEATURE_DIR, "X_mel.npy"))
        x2 = np.load(os.path.join(FEATURE_DIR, "X_mfcc.npy"))
        return {"Mel": x1.shape[1:], "MFCC": x2.shape[1:]}
    x = np.load(os.path.join(FEATURE_DIR, f"{feature_name}.npy"))
    return x.shape[1:]


def build_blocks(model_name, feature_name):
    if feature_name == "Fusion_Mel_MFCC":
        return [
            ("Mel Input", "Mel\n(Freq x Time)"),
            ("MFCC Input", "MFCC\n(Freq x Time)"),
            ("Mel Branch", "Conv-BN-ReLU\nMaxPool-Conv-GAP"),
            ("MFCC Branch", "Conv-BN-ReLU\nMaxPool-Conv-GAP"),
            ("Fusion", "Concat"),
            ("Classifier", "FC-Dropout-FC"),
            ("Output", "4-class logits"),
        ]

    if model_name == "SimpleCNN":
        return [
            ("Input", f"{feature_name}\n(Freq x Time)"),
            ("Block 1", "Conv-BN-ReLU-Pool"),
            ("Block 2", "Conv-BN-ReLU-Pool"),
            ("Block 3", "Conv-BN-ReLU-Pool"),
            ("Pooling", "Global Avg Pool"),
            ("Classifier", "FC"),
            ("Output", "4-class logits"),
        ]

    if model_name.startswith("ResNet"):
        return [
            ("Input", f"{feature_name}\n(Freq x Time)"),
            ("Stem", "7x7 Conv + Pool"),
            ("Stage 1", "Residual Blocks"),
            ("Stage 2", "Residual Blocks"),
            ("Stage 3", "Residual Blocks"),
            ("Stage 4", "Residual Blocks"),
            ("Classifier", "Global Pool + FC"),
            ("Output", "4-class logits"),
        ]

    if model_name == "RNN":
        return [
            ("Input", f"{feature_name}\n(Time x Freq)"),
            ("Reshape", "Transpose to sequence"),
            ("Encoder", "Bi-RNN"),
            ("Temporal Readout", "Last time step"),
            ("Classifier", "FC"),
            ("Output", "4-class logits"),
        ]

    if model_name == "LSTM":
        return [
            ("Input", f"{feature_name}\n(Time x Freq)"),
            ("Reshape", "Transpose to sequence"),
            ("Encoder", "Bi-LSTM"),
            ("Temporal Readout", "Last time step"),
            ("Classifier", "FC"),
            ("Output", "4-class logits"),
        ]

    if model_name == "Transformer":
        return [
            ("Input", f"{feature_name}\n(Time x Freq)"),
            ("Embedding", "Linear to d_model"),
            ("Position", "Learnable Positional Encoding"),
            ("Encoder", "Transformer Encoder"),
            ("Aggregation", "Temporal Mean Pooling"),
            ("Classifier", "FC"),
            ("Output", "4-class logits"),
        ]

    return [
        ("Input", feature_name),
        ("Model", model_name),
        ("Output", "4-class logits"),
    ]


def draw_single_path_architecture(model_name, feature_name, feature_shape, save_stem):
    blocks = build_blocks(model_name, feature_name)
    fig, ax = plt.subplots(figsize=(16, 3.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    n_blocks = len(blocks)
    width = 0.11 if n_blocks >= 7 else 0.13
    height = 0.32
    x_positions = np.linspace(0.02, 0.88, n_blocks)
    y = 0.34

    title = f"Architecture Diagram: {model_name} + {feature_name}"
    shape_text = f"Input shape: {feature_shape[0]} x {feature_shape[1]}" if isinstance(feature_shape, tuple) else ""
    ax.text(0.5, 0.94, title, ha="center", va="center", fontsize=15, fontweight="bold")
    if shape_text:
        ax.text(0.5, 0.86, shape_text, ha="center", va="center", fontsize=10)

    for idx, (_, text) in enumerate(blocks):
        add_box(ax, (x_positions[idx], y), width, height, text)
        if idx < n_blocks - 1:
            add_arrow(ax, (x_positions[idx] + width, y + height / 2), (x_positions[idx + 1], y + height / 2))

    save_current_figure(save_stem)


def draw_fusion_architecture(save_stem, fusion_shape):
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.95, "Architecture Diagram: DualStreamFusionModel + Fusion_Mel_MFCC", ha="center", va="center", fontsize=15, fontweight="bold")
    ax.text(
        0.5,
        0.89,
        f"Mel input: {fusion_shape['Mel'][0]} x {fusion_shape['Mel'][1]}    MFCC input: {fusion_shape['MFCC'][0]} x {fusion_shape['MFCC'][1]}",
        ha="center",
        va="center",
        fontsize=10,
    )

    add_box(ax, (0.06, 0.62), 0.16, 0.16, "Mel Input")
    add_box(ax, (0.30, 0.62), 0.20, 0.16, "Mel Branch\nConv-BN-ReLU\nMaxPool-Conv-GAP")
    add_box(ax, (0.06, 0.22), 0.16, 0.16, "MFCC Input")
    add_box(ax, (0.30, 0.22), 0.20, 0.16, "MFCC Branch\nConv-BN-ReLU\nMaxPool-Conv-GAP")
    add_box(ax, (0.60, 0.42), 0.14, 0.18, "Feature\nConcat")
    add_box(ax, (0.82, 0.42), 0.12, 0.18, "FC-\nDropout-\nFC")

    add_arrow(ax, (0.22, 0.70), (0.30, 0.70))
    add_arrow(ax, (0.22, 0.30), (0.30, 0.30))
    add_arrow(ax, (0.50, 0.70), (0.60, 0.51))
    add_arrow(ax, (0.50, 0.30), (0.60, 0.51))
    add_arrow(ax, (0.74, 0.51), (0.82, 0.51))
    add_arrow(ax, (0.94, 0.51), (0.99, 0.51))
    ax.text(0.995, 0.51, "4-class logits", ha="left", va="center", fontsize=10)

    save_current_figure(save_stem)


def generate_all_architecture_diagrams():
    records = []
    for feature_name in SINGLE_FEATURES:
        feature_shape = get_feature_shape(feature_name)
        for model_name in SINGLE_MODELS:
            save_stem = os.path.join(ARCH_DIR, f"arch_{model_name}_{feature_name}")
            draw_single_path_architecture(model_name, feature_name, feature_shape, save_stem)
            records.append({
                "Model": model_name,
                "Feature": feature_name,
                "FigureStem": save_stem,
            })

    fusion_shape = get_feature_shape("Fusion_Mel_MFCC")
    fusion_stem = os.path.join(ARCH_DIR, "arch_DualStreamFusionModel_Fusion_Mel_MFCC")
    draw_fusion_architecture(fusion_stem, fusion_shape)
    records.append({
        "Model": "DualStreamFusionModel",
        "Feature": "Fusion_Mel_MFCC",
        "FigureStem": fusion_stem,
    })

    index_csv = os.path.join(OUTPUT_DIR, "model_architecture_index.csv")
    pd.DataFrame(records).to_csv(index_csv, index=False)


def draw_workflow_diagram():
    fig, ax = plt.subplots(figsize=(18, 5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.93, "Overall Experimental Workflow", ha="center", va="center", fontsize=16, fontweight="bold")

    boxes = [
        ((0.02, 0.36), 0.12, 0.22, "DeepShip\nRaw Audio"),
        ((0.17, 0.36), 0.14, 0.22, "Preprocess\n32kHz / Segment /\nBand-pass"),
        ((0.36, 0.36), 0.14, 0.22, "Feature\nExtraction\nSTFT/Mel/MFCC/CQT"),
        ((0.55, 0.36), 0.12, 0.22, "Dimensionality\nReduction\nPCA / t-SNE"),
        ((0.72, 0.36), 0.12, 0.22, "Model\nTraining\n7 models + fusion"),
        ((0.87, 0.36), 0.11, 0.22, "Evaluation\n& Analysis"),
    ]

    for xy, width, height, text in boxes:
        add_box(ax, xy, width, height, text, fontsize=11)

    for idx in range(len(boxes) - 1):
        start = (boxes[idx][0][0] + boxes[idx][1], boxes[idx][0][1] + boxes[idx][2] / 2)
        end = (boxes[idx + 1][0][0], boxes[idx + 1][0][1] + boxes[idx + 1][2] / 2)
        add_arrow(ax, start, end)

    ax.text(0.93, 0.18, "Outputs:\nTraining logs / PR curves /\nConfusion matrices /\nGlobal charts / Robustness", ha="center", va="center", fontsize=10)
    save_current_figure(os.path.join(OUTPUT_DIR, "workflow_overview"))


def create_feature_montage():
    image_files = [
        os.path.join(FEATURE_DIR, "comparison_Cargo.png"),
        os.path.join(FEATURE_DIR, "comparison_Tanker.png"),
        os.path.join(FEATURE_DIR, "comparison_Tug.png"),
        os.path.join(FEATURE_DIR, "comparison_Passengership.png"),
    ]
    titles = ["Cargo", "Tanker", "Tug", "Passengership"]
    existing = [(img, title) for img, title in zip(image_files, titles) if os.path.exists(img)]
    if not existing:
        return

    create_image_grid(
        [item[0] for item in existing],
        [item[1] for item in existing],
        os.path.join(OUTPUT_DIR, "feature_comparison_montage"),
        ncols=2,
        figsize=(16, 12),
        suptitle="Key Sample Time-Frequency Comparison Montage",
    )


def create_dimensionality_reduction_montage():
    image_paths = [
        os.path.join(DIM_REDUCTION_DIR, "dr_X_mel_full.png"),
        os.path.join(DIM_REDUCTION_DIR, "dr_X_mfcc_full.png"),
        os.path.join(DIM_REDUCTION_DIR, "dr_X_cqt_full.png"),
        os.path.join(DIM_REDUCTION_DIR, "dr_X_stft_full.png"),
    ]
    titles = [
        "Mel: PCA + t-SNE",
        "MFCC: PCA + t-SNE",
        "CQT: PCA + t-SNE",
        "STFT: PCA + t-SNE",
    ]
    create_image_grid(
        image_paths,
        titles,
        os.path.join(OUTPUT_DIR, "dimensionality_reduction_montage"),
        ncols=2,
        figsize=(16, 12),
        suptitle="Dimensionality Reduction Results Montage",
    )


def group_result_images(prefix):
    files = [f for f in os.listdir(RESULTS_DIR) if f.startswith(prefix) and f.endswith(".png")]
    files.sort()
    groups = {}
    for filename in files:
        stem = filename[:-4]
        suffix = stem[len(prefix):]
        if suffix.startswith("fusion_Mel_MFCC"):
            feature = "Fusion_Mel_MFCC"
            model = "DualStreamFusionModel"
        else:
            parts = suffix.split("_")
            if len(parts) < 3:
                continue
            model = parts[0]
            feature = "_".join(parts[1:])
        groups.setdefault(feature, []).append((model, os.path.join(RESULTS_DIR, filename)))
    return groups


def create_result_montages(prefix, output_name_prefix, figure_title_prefix):
    groups = group_result_images(prefix)
    index_rows = []
    for feature, items in groups.items():
        items.sort(key=lambda x: x[0])
        image_paths = [path for _, path in items]
        titles = [model for model, _ in items]
        save_stem = os.path.join(OUTPUT_DIR, f"{output_name_prefix}_{feature}")
        created = create_image_grid(
            image_paths,
            titles,
            save_stem,
            ncols=4,
            figsize=(20, 12),
            suptitle=f"{figure_title_prefix}: {feature}",
        )
        for path in created:
            index_rows.append({"Category": output_name_prefix, "Feature": feature, "Path": path})
    return index_rows


def copy_existing_global_figures():
    stems = [
        "global_f1_barplot",
        "global_f1_heatmap",
        "global_metric_model_feature_heatmaps",
        "model_parameters_comparison",
        "model_inference_time_comparison",
        "robustness_curve_all_models",
        "robustness_heatmap_all_models",
    ]
    copied_rows = []
    for stem in stems:
        src_stem = os.path.join(RESULTS_DIR, stem)
        dst_stem = os.path.join(OUTPUT_DIR, stem)
        copied = copy_all_formats(src_stem, dst_stem)
        for path in copied:
            copied_rows.append({"Category": "global_or_summary", "Feature": "", "Path": path})
    return copied_rows


def main():
    ensure_dir(OUTPUT_DIR)
    draw_workflow_diagram()
    create_feature_montage()
    create_dimensionality_reduction_montage()
    generate_all_architecture_diagrams()
    index_rows = []
    index_rows.extend(create_result_montages("cm_", "cm_montage", "Confusion Matrix Montage"))
    index_rows.extend(create_result_montages("history_", "history_montage", "Training History Montage"))
    index_rows.extend(create_result_montages("pr_curve_", "pr_curve_montage", "PR Curve Montage"))
    index_rows.extend(copy_existing_global_figures())
    if index_rows:
        pd.DataFrame(index_rows).to_csv(os.path.join(OUTPUT_DIR, "paper_figures_index.csv"), index=False)
    print(f"Paper figures will be saved under {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
