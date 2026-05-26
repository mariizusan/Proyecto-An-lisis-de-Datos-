"""
dataset_viz.py — Visualización de la composición del dataset
y separación de splits train/val/test.

Uso:
    from dataset_viz import visualizar_dataset, visualizar_splits
    visualizar_dataset()
    visualizar_splits()
"""

import os
import pickle
import hashlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
import configl

TROPE_NAMES  = {0: "hurt_comfort", 1: "fluff", 2: "slow_burn"}
TROPE_COLORS = {0: "#E24B4A", 1: "#1D9E75", 2: "#7B68D8"}
FANDOM_COLORS = {
    "One Direction": "#F4A261",
    "BTS":           "#2E86AB",
    "Harry Potter":  "#7B3F00",
}

def _get_split(title: str) -> str:
    hash_val = int(hashlib.md5(title.encode()).hexdigest(), 16) % 100
    return "train" if hash_val < 70 else "val" if hash_val < 85 else "test"


def cargar_df(pkl_path: str = "data/features/emotional_features.pkl") -> pd.DataFrame:
    """Carga el pkl y lo convierte en DataFrame con metadatos por fanfic."""
    with open(pkl_path, "rb") as f:
        features = pickle.load(f)

    rows = []
    for f in features:
        trope_idx = f["trope"] if isinstance(f["trope"], int) else \
                    configl.TROPE_TO_IDX.get(f["trope"], -1)
        n_caps    = f["num_chapters"]
        words     = sum(
            len(ch.split()) if isinstance(ch, str) else ch.shape[0] * 400
            for ch in (f.get("chapters") or
                       [None] * n_caps)
        ) if f.get("chapters") else n_caps * 3000

        # Palabras estimadas desde embeddings si no hay texto
        total_windows = sum(
            emb.shape[0] for emb in f["chapter_embeddings"]
        ) if "chapter_embeddings" in f else 0

        rows.append({
            "title":        f.get("title", ""),
            "fandom":       f.get("fandom", ""),
            "trope_idx":    trope_idx,
            "trope":        TROPE_NAMES.get(trope_idx, str(trope_idx)),
            "n_chapters":   n_caps,
            "n_windows":    total_windows,
            "words_est":    total_windows * 400,   # ~400 palabras por ventana
            "is_oneshot":   n_caps == 1,
            "split":        _get_split(f.get("title", "")),
        })

    return pd.DataFrame(rows)


def visualizar_dataset(
    pkl_path: str = "data/features/emotional_features.pkl",
    save: bool = True,
):
    """
    Figura de 6 paneles que muestra la composición del dataset:
      A) Conteo por fandom × tropo (heatmap)
      B) Distribución de capítulos por tropo (boxplot)
      C) Distribución de palabras estimadas por tropo (violín)
      D) Proporción oneshot vs multi-capítulo por tropo
      E) Conteo total por fandom
      F) Conteo total por tropo
    """
    df = cargar_df(pkl_path)
    print(f"Dataset: {len(df)} fanfics")
    print(df.groupby(["fandom", "trope"]).size().to_string())

    fig = plt.figure(figsize=(20, 14))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── A) Heatmap fandom × tropo ──
    ax_heat = fig.add_subplot(gs[0, :2])
    pivot = df.groupby(["fandom", "trope"]).size().unstack(fill_value=0)
    # Ordenar tropos
    for t in TROPE_NAMES.values():
        if t not in pivot.columns:
            pivot[t] = 0
    pivot = pivot[[t for t in TROPE_NAMES.values() if t in pivot.columns]]

    sns.heatmap(pivot, annot=True, fmt="d", cmap="YlOrRd",
                linewidths=0.5, ax=ax_heat,
                cbar_kws={"label": "nº fanfics"})
    ax_heat.set_title("Composición del dataset: fandom × tropo",
                      fontweight="bold", fontsize=11)
    ax_heat.set_xlabel("Tropo"); ax_heat.set_ylabel("Fandom")
    ax_heat.tick_params(axis="x", rotation=15)

    # ── B) Distribución de capítulos por tropo ──
    ax_caps = fig.add_subplot(gs[0, 2])
    data_caps = [df[df["trope"] == t]["n_chapters"].values
                 for t in TROPE_NAMES.values()]
    bp = ax_caps.boxplot(data_caps, patch_artist=True,
                         labels=list(TROPE_NAMES.values()),
                         showfliers=True, widths=0.5)
    for patch, t_idx in zip(bp["boxes"], range(3)):
        patch.set_facecolor(TROPE_COLORS[t_idx])
        patch.set_alpha(0.7)
    ax_caps.set_title("Distribución de capítulos\npor tropo", fontweight="bold")
    ax_caps.set_ylabel("Número de capítulos")
    ax_caps.tick_params(axis="x", rotation=15, labelsize=8)
    ax_caps.grid(alpha=0.3, axis="y")

    # ── C) Palabras estimadas por tropo (violín) ──
    ax_words = fig.add_subplot(gs[1, 0])
    trope_order = list(TROPE_NAMES.values())
    parts = ax_words.violinplot(
        [df[df["trope"] == t]["words_est"].values for t in trope_order],
        positions=range(len(trope_order)),
        showmedians=True, showextrema=True,
    )
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(TROPE_COLORS[i])
        pc.set_alpha(0.6)
    ax_words.set_xticks(range(len(trope_order)))
    ax_words.set_xticklabels(trope_order, rotation=15, fontsize=8)
    ax_words.set_title("Palabras estimadas por tropo\n(~400 palabras/ventana)",
                       fontweight="bold")
    ax_words.set_ylabel("Palabras estimadas")
    ax_words.grid(alpha=0.3, axis="y")

    # ── D) Oneshot vs multi-capítulo por tropo ──
    ax_one = fig.add_subplot(gs[1, 1])
    for i, t in enumerate(TROPE_NAMES.values()):
        subset   = df[df["trope"] == t]
        n_one    = subset["is_oneshot"].sum()
        n_multi  = len(subset) - n_one
        ax_one.bar(i - 0.2, n_one,  width=0.38, color=TROPE_COLORS[i],
                   alpha=0.9, label="oneshot" if i == 0 else "")
        ax_one.bar(i + 0.2, n_multi, width=0.38, color=TROPE_COLORS[i],
                   alpha=0.4, label="multi-cap" if i == 0 else "")
        ax_one.text(i - 0.2, n_one  + 0.2, str(n_one),
                    ha="center", fontsize=8)
        ax_one.text(i + 0.2, n_multi + 0.2, str(n_multi),
                    ha="center", fontsize=8)
    ax_one.set_xticks(range(len(trope_order)))
    ax_one.set_xticklabels(trope_order, rotation=15, fontsize=8)
    ax_one.set_title("Oneshot vs multi-capítulo\npor tropo", fontweight="bold")
    ax_one.set_ylabel("Número de fanfics")
    ax_one.legend(fontsize=8); ax_one.grid(alpha=0.3, axis="y")

    # ── E) Conteo por fandom ──
    ax_fandom = fig.add_subplot(gs[1, 2])
    fandom_counts = df.groupby("fandom").size().sort_values(ascending=True)
    colors_f = [FANDOM_COLORS.get(f, "#888") for f in fandom_counts.index]
    ax_fandom.barh(fandom_counts.index, fandom_counts.values,
                   color=colors_f, alpha=0.85, edgecolor="white")
    for i, v in enumerate(fandom_counts.values):
        ax_fandom.text(v + 0.3, i, str(v), va="center", fontsize=9)
    ax_fandom.set_title("Fanfics por fandom", fontweight="bold")
    ax_fandom.set_xlabel("Número de fanfics")
    ax_fandom.grid(alpha=0.3, axis="x")

    # ── F) Ventanas por fandom × tropo (muestra la riqueza de secuencia) ──
    ax_wins = fig.add_subplot(gs[2, :2])
    pivot_wins = df.groupby(["fandom", "trope"])["n_windows"].mean().unstack(fill_value=0)
    for t in TROPE_NAMES.values():
        if t not in pivot_wins.columns:
            pivot_wins[t] = 0
    pivot_wins = pivot_wins[[t for t in TROPE_NAMES.values()
                              if t in pivot_wins.columns]]
    sns.heatmap(pivot_wins, annot=True, fmt=".0f", cmap="Blues",
                linewidths=0.5, ax=ax_wins,
                cbar_kws={"label": "ventanas promedio"})
    ax_wins.set_title("Ventanas promedio por fandom × tropo\n"
                      "(proxy de longitud media del fanfic)",
                      fontweight="bold")
    ax_wins.set_xlabel("Tropo"); ax_wins.set_ylabel("Fandom")
    ax_wins.tick_params(axis="x", rotation=15)

    # ── G) Balance de capítulos — histograma general ──
    ax_hist = fig.add_subplot(gs[2, 2])
    for t_idx, t in TROPE_NAMES.items():
        vals = df[df["trope"] == t]["n_chapters"].values
        ax_hist.hist(vals, bins=range(1, 17), alpha=0.6,
                     color=TROPE_COLORS[t_idx], label=t,
                     edgecolor="white")
    ax_hist.set_title("Histograma de capítulos", fontweight="bold")
    ax_hist.set_xlabel("Número de capítulos")
    ax_hist.set_ylabel("Frecuencia")
    ax_hist.legend(fontsize=7); ax_hist.grid(alpha=0.3, axis="y")

    plt.suptitle("Composición del Dataset de Fanfiction",
                 fontsize=14, fontweight="bold", y=1.01)
    os.makedirs(configl.FIGURES_DIR, exist_ok=True)
    if save:
        plt.savefig(f"{configl.FIGURES_DIR}10_dataset_composicion.png",
                    dpi=150, bbox_inches="tight", facecolor="white")
    plt.show()
    return df


def visualizar_splits(
    pkl_path: str = "data/features/emotional_features.pkl",
    save: bool = True,
):
    """
    Figura de 3 paneles mostrando cómo quedaron los splits:
      A) Conteo por split × tropo
      B) Conteo por split × fandom
      C) Distribución de capítulos por split
    """
    df = cargar_df(pkl_path)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    split_colors = {"train": "#2E86AB", "val": "#F4A261", "test": "#E24B4A"}
    splits = ["train", "val", "test"]

    # ── A) Split × tropo ──
    ax = axes[0]
    pivot = df.groupby(["split", "trope"]).size().unstack(fill_value=0)
    pivot = pivot.reindex(splits)
    for t in TROPE_NAMES.values():
        if t not in pivot.columns:
            pivot[t] = 0
    pivot = pivot[[t for t in TROPE_NAMES.values() if t in pivot.columns]]

    x      = np.arange(len(splits))
    width  = 0.25
    for i, t in enumerate(TROPE_NAMES.values()):
        vals = pivot[t].values if t in pivot.columns else np.zeros(3)
        bars = ax.bar(x + i * width, vals, width,
                      label=t, color=TROPE_COLORS[i], alpha=0.85,
                      edgecolor="white")
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width()/2, v + 0.2,
                        str(int(v)), ha="center", fontsize=8)

    ax.set_xticks(x + width)
    ax.set_xticklabels(splits)
    ax.set_title("Distribución de tropos por split", fontweight="bold")
    ax.set_ylabel("Número de fanfics")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

    # Añadir totales por split
    totals = df.groupby("split").size().reindex(splits)
    for i, (split, total) in enumerate(totals.items()):
        ax.text(i + width, ax.get_ylim()[1] * 0.95,
                f"n={total}", ha="center", fontsize=9,
                fontweight="bold",
                color=split_colors[split])

    # ── B) Split × fandom ──
    ax2 = axes[1]
    pivot_f = df.groupby(["split", "fandom"]).size().unstack(fill_value=0)
    pivot_f = pivot_f.reindex(splits)
    fandoms = list(pivot_f.columns)

    width_f = 0.8 / len(fandoms)
    for i, fandom in enumerate(fandoms):
        vals = pivot_f[fandom].fillna(0).values
        bars = ax2.bar(x + i * width_f, vals, width_f,
                       label=fandom,
                       color=FANDOM_COLORS.get(fandom, "#888"),
                       alpha=0.85, edgecolor="white")
        for bar, v in zip(bars, vals):
            if v > 0:
                ax2.text(bar.get_x() + bar.get_width()/2, v + 0.2,
                         str(int(v)), ha="center", fontsize=7)

    ax2.set_xticks(x + width_f * len(fandoms) / 2)
    ax2.set_xticklabels(splits)
    ax2.set_title("Distribución de fandoms por split", fontweight="bold")
    ax2.set_ylabel("Número de fanfics")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3, axis="y")

    # ── C) Capítulos por split ──
    ax3 = axes[2]
    data_splits = [df[df["split"] == s]["n_chapters"].values for s in splits]
    bp = ax3.boxplot(data_splits, patch_artist=True,
                     labels=splits, widths=0.5, showfliers=True)
    for patch, s in zip(bp["boxes"], splits):
        patch.set_facecolor(split_colors[s])
        patch.set_alpha(0.7)

    # Superponer puntos individuales
    for i, (s, data) in enumerate(zip(splits, data_splits)):
        jitter = np.random.normal(0, 0.06, len(data))
        ax3.scatter(np.full(len(data), i + 1) + jitter, data,
                    alpha=0.4, s=15, color=split_colors[s], zorder=3)

    ax3.set_title("Distribución de capítulos\npor split", fontweight="bold")
    ax3.set_ylabel("Número de capítulos")
    ax3.grid(alpha=0.3, axis="y")

    # Tabla resumen debajo
    print("\nResumen de splits:")
    print(f"{'Split':<8} {'Total':>7} {'HC':>5} {'Fluff':>7} {'SB':>8} "
          f"{'Caps avg':>10} {'Oneshots':>10}")
    print("-" * 60)
    for s in splits:
        sub = df[df["split"] == s]
        hc  = len(sub[sub["trope"] == "hurt_comfort"])
        fl  = len(sub[sub["trope"] == "fluff"])
        sb  = len(sub[sub["trope"] == "slow_burn"])
        avg = sub["n_chapters"].mean()
        one = sub["is_oneshot"].sum()
        print(f"{s:<8} {len(sub):>7} {hc:>5} {fl:>7} {sb:>8} "
              f"{avg:>10.1f} {one:>10}")

    plt.suptitle("Separación Train / Val / Test",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    os.makedirs(configl.FIGURES_DIR, exist_ok=True)
    if save:
        plt.savefig(f"{configl.FIGURES_DIR}11_splits.png",
                    dpi=150, bbox_inches="tight", facecolor="white")
    plt.show()
    return df


def pca_ventanas_roberta(
    pkl_path: str = "data/features/emotional_features.pkl",
    tropo_filter: int = None,
    max_ventanas: int = 500,
    save: bool = True,
):
    """
    PCA de los embeddings crudos de RoBERTa por ventana.
    Cada punto = una ventana de 512 tokens de un capítulo.

    Más informativo que PCA de hidden states para entender
    qué estructura emocional captura RoBERTa antes de la LSTM.

    Parámetros:
        tropo_filter: si int, solo usa ese tropo (None = todos)
        max_ventanas: límite de ventanas para que el plot no se sature
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    with open(pkl_path, "rb") as f:
        features = pickle.load(f)

    all_vecs, all_tropes, all_fandoms, all_caps_pos = [], [], [], []

    for f in features:
        t_idx = f["trope"] if isinstance(f["trope"], int) else \
                configl.TROPE_TO_IDX.get(f["trope"], -1)
        if tropo_filter is not None and t_idx != tropo_filter:
            continue

        n_caps = f["num_chapters"]
        for c_idx, emb in enumerate(f["chapter_embeddings"]):
            emb = np.array(emb)
            for w_idx in range(emb.shape[0]):
                all_vecs.append(emb[w_idx])
                all_tropes.append(t_idx)
                all_fandoms.append(f.get("fandom", ""))
                # Posición relativa dentro de la obra (0=inicio, 1=fin)
                all_caps_pos.append(c_idx / max(n_caps - 1, 1))

    X       = np.stack(all_vecs)
    tropes  = np.array(all_tropes)
    pos     = np.array(all_caps_pos)

    # Subsamplear si hay demasiadas ventanas
    if len(X) > max_ventanas:
        idx = np.random.choice(len(X), max_ventanas, replace=False)
        X, tropes, pos = X[idx], tropes[idx], pos[idx]

    print(f"PCA sobre {len(X)} ventanas de 768-dim...")

    pca   = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    var   = pca.explained_variance_ratio_

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # ── PCA por tropo ──
    ax = axes[0]
    for t_idx in range(3):
        if tropo_filter is not None and t_idx != tropo_filter:
            continue
        mask = tropes == t_idx
        if mask.sum() == 0:
            continue
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=TROPE_COLORS[t_idx],
                   label=TROPE_NAMES[t_idx],
                   alpha=0.5, s=20, edgecolors="none")
    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}% var)")
    ax.set_title("PCA de embeddings RoBERTa por ventana\ncoloreado por tropo",
                 fontweight="bold")
    ax.legend(); ax.grid(alpha=0.2)

    # ── PCA por posición narrativa ──
    ax2 = axes[1]
    sc = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=pos,
                     cmap="plasma", alpha=0.5, s=20, edgecolors="none")
    plt.colorbar(sc, ax=ax2, label="Posición relativa (0=inicio, 1=fin)")
    ax2.set_xlabel(f"PC1 ({var[0]*100:.1f}% var)")
    ax2.set_ylabel(f"PC2 ({var[1]*100:.1f}% var)")
    ax2.set_title("PCA de embeddings RoBERTa por ventana\ncoloreado por posición narrativa",
                  fontweight="bold")
    ax2.grid(alpha=0.2)

    tropo_str = TROPE_NAMES.get(tropo_filter, "todos") \
                if tropo_filter is not None else "todos los tropos"
    plt.suptitle(f"Espacio de representación RoBERTa — {tropo_str}",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()

    os.makedirs(configl.FIGURES_DIR, exist_ok=True)
    if save:
        fname = f"12_pca_ventanas_{tropo_str.replace(' ', '_')}.png"
        plt.savefig(f"{configl.FIGURES_DIR}{fname}",
                    dpi=150, bbox_inches="tight", facecolor="white")
    plt.show()


if __name__ == "__main__":
    visualizar_dataset()
    visualizar_splits()
    pca_ventanas_roberta()
