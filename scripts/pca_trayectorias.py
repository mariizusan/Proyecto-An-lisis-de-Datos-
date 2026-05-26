"""
pca_trayectorias.py — PCA con trayectorias por fanfic

En lugar de scatter de todas las ventanas mezcladas,
plotea la trayectoria de cada fanfic como una línea en el espacio PCA.
Cada línea = un fanfic. Punto inicial = inicio del fanfic, punto final = fin.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import configl

TROPE_NAMES  = {0: "hurt_comfort", 1: "fluff", 2: "slow_burn"}
TROPE_COLORS = {0: "#E24B4A", 1: "#1D9E75", 2: "#7B68D8"}


def _cargar_secuencias(
    pkl_path: str,
    tropo_filter: int = None,
    min_ventanas: int = 3,
) -> list:
    """
    Carga secuencias de embeddings por fanfic.
    Cada elemento: dict con {embeddings: (n_wins_total, 768), trope, title, fandom}
    Concatena todas las ventanas de todos los capítulos en orden.
    """
    with open(pkl_path, "rb") as f:
        features = pickle.load(f)

    secuencias = []
    for f in features:
        t_idx = f["trope"] if isinstance(f["trope"], int) else \
                configl.TROPE_TO_IDX.get(f["trope"], -1)
        if tropo_filter is not None and t_idx != tropo_filter:
            continue

        # Concatenar ventanas de todos los capítulos en orden
        all_windows = np.concatenate(
            [np.array(emb) for emb in f["chapter_embeddings"]], axis=0
        )   # (n_ventanas_total, 768)

        if all_windows.shape[0] < min_ventanas:
            continue

        secuencias.append({
            "embeddings": all_windows,
            "trope":      t_idx,
            "title":      f.get("title", ""),
            "fandom":     f.get("fandom", ""),
            "n_chapters": f["num_chapters"],
            "n_windows":  all_windows.shape[0],
        })

    return secuencias


def pca_trayectorias(
    pkl_path: str = "data/features/emotional_features.pkl",
    tropo_filter: int = None,
    n_components_check: int = 30,
    max_fanfics_por_tropo: int = 10,
    save: bool = True,
):
    """
    Figura de 4 paneles:
      A) Varianza explicada acumulada (cuántos PCs necesitas)
      B) Trayectorias en PC1-PC2 coloreadas por tropo
      C) Trayectorias en PC1-PC2 coloreadas por posición (inicio→fin)
      D) Proyección en PC1 vs posición — ¿hay tendencia lineal?
    """
    secuencias = _cargar_secuencias(pkl_path, tropo_filter)
    if not secuencias:
        print("No hay datos suficientes.")
        return

    # Ajustar PCA sobre TODAS las ventanas juntas
    all_X = np.concatenate([s["embeddings"] for s in secuencias], axis=0)
    print(f"Ajustando PCA sobre {len(all_X)} ventanas de {all_X.shape[1]}-dim...")

    scaler  = StandardScaler()
    all_X_s = scaler.fit_transform(all_X)

    pca_full = PCA(n_components=min(n_components_check, all_X.shape[0],
                                    all_X.shape[1]), random_state=42)
    pca_full.fit(all_X_s)
    var_cum = np.cumsum(pca_full.explained_variance_ratio_)

    pca2    = PCA(n_components=2, random_state=42)
    pca2.fit(all_X_s)

    fig = plt.figure(figsize=(20, 16))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # ── A) Varianza explicada acumulada ───────────────────────────────────────
    ax_var = fig.add_subplot(gs[0, 0])
    n_pcs  = len(var_cum)
    ax_var.plot(range(1, n_pcs + 1), var_cum, "-o",
                color="#2E86AB", linewidth=2, markersize=4)
    for thresh, color, label in [(0.5, "#F4A261", "50%"),
                                  (0.7, "#E24B4A", "70%"),
                                  (0.9, "#7B68D8", "90%")]:
        ax_var.axhline(thresh, color=color, linestyle="--",
                       alpha=0.7, label=f"{label} var")
        idx = np.searchsorted(var_cum, thresh)
        if idx < n_pcs:
            ax_var.annotate(f"{idx+1} PCs",
                            xy=(idx + 1, thresh),
                            xytext=(idx + 3, thresh - 0.03),
                            fontsize=8, color=color,
                            arrowprops=dict(arrowstyle="->",
                                            color=color, lw=0.8))
    ax_var.set_xlabel("Número de componentes principales")
    ax_var.set_ylabel("Varianza acumulada explicada")
    ax_var.set_title("¿Cuántos PCs necesitas?\n"
                     "Varianza acumulada de embeddings RoBERTa",
                     fontweight="bold")
    ax_var.legend(fontsize=8); ax_var.grid(alpha=0.3)
    ax_var.set_xlim(1, n_pcs)

    # ── B) Trayectorias coloreadas por tropo ─────────────────────────────────
    ax_traj = fig.add_subplot(gs[0, 1])

    # Subsamplear fanfics para no saturar el plot
    import random
    random.seed(42)
    fanfics_por_tropo = {}
    for s in secuencias:
        t = s["trope"]
        fanfics_por_tropo.setdefault(t, []).append(s)
    muestra = []
    for t, flist in fanfics_por_tropo.items():
        muestra.extend(random.sample(flist, min(max_fanfics_por_tropo, len(flist))))

    for s in muestra:
        emb_s  = scaler.transform(s["embeddings"])
        coords = pca2.transform(emb_s)   # (n_wins, 2)
        color  = TROPE_COLORS[s["trope"]]

        # Línea con transparencia
        ax_traj.plot(coords[:, 0], coords[:, 1],
                     "-", color=color, alpha=0.35, linewidth=1.2)
        # Punto de inicio
        ax_traj.scatter(coords[0, 0], coords[0, 1],
                        color=color, s=40, zorder=5,
                        marker="o", edgecolors="white", linewidths=0.5)
        # Punto final con flecha
        if len(coords) > 1:
            dx = coords[-1, 0] - coords[-2, 0]
            dy = coords[-1, 1] - coords[-2, 1]
            ax_traj.annotate("",
                xy=(coords[-1, 0], coords[-1, 1]),
                xytext=(coords[-2, 0], coords[-2, 1]),
                arrowprops=dict(arrowstyle="->", color=color,
                                lw=1.2, alpha=0.7)
            )

    # Leyenda manual
    for t_idx, t_name in TROPE_NAMES.items():
        if tropo_filter is not None and t_idx != tropo_filter:
            continue
        ax_traj.plot([], [], "-o", color=TROPE_COLORS[t_idx],
                     label=t_name, linewidth=1.5)
    ax_traj.legend(fontsize=8)
    ax_traj.set_xlabel(f"PC1 ({pca2.explained_variance_ratio_[0]*100:.1f}% var)")
    ax_traj.set_ylabel(f"PC2 ({pca2.explained_variance_ratio_[1]*100:.1f}% var)")
    ax_traj.set_title("Trayectorias en espacio PCA\n"
                      "○=inicio → →=fin  (cada línea = un fanfic)",
                      fontweight="bold")
    ax_traj.grid(alpha=0.2)

    # ── C) Trayectorias coloreadas por posición (gradiente) ──────────────────
    ax_grad = fig.add_subplot(gs[1, 0])

    for s in muestra:
        emb_s  = scaler.transform(s["embeddings"])
        coords = pca2.transform(emb_s)
        n      = len(coords)
        pos    = np.linspace(0, 1, n)

        # Segmentos con color gradiente inicio→fin
        points  = coords.reshape(-1, 1, 2)
        segs    = np.concatenate([points[:-1], points[1:]], axis=1)
        lc      = LineCollection(segs, cmap="plasma",
                                 norm=plt.Normalize(0, 1),
                                 alpha=0.5, linewidth=1.5)
        lc.set_array(pos[:-1])
        ax_grad.add_collection(lc)

        ax_grad.scatter(coords[0, 0], coords[0, 1],
                        color="purple", s=25, zorder=5, alpha=0.6)
        ax_grad.scatter(coords[-1, 0], coords[-1, 1],
                        color="yellow", s=25, zorder=5, alpha=0.6,
                        edgecolors="gray", linewidths=0.3)

    ax_grad.autoscale()
    sm = plt.cm.ScalarMappable(cmap="plasma", norm=plt.Normalize(0, 1))
    sm.set_array([])
    plt.colorbar(sm, ax=ax_grad, label="Posición narrativa (0=inicio, 1=fin)",
                 fraction=0.03)
    ax_grad.set_xlabel(f"PC1 ({pca2.explained_variance_ratio_[0]*100:.1f}% var)")
    ax_grad.set_ylabel(f"PC2 ({pca2.explained_variance_ratio_[1]*100:.1f}% var)")
    ax_grad.set_title("Trayectorias coloreadas por posición narrativa\n"
                      "●=inicio (morado) → ●=fin (amarillo)",
                      fontweight="bold")
    ax_grad.grid(alpha=0.2)

    # ── D) PC1 vs posición — ¿hay tendencia? ─────────────────────────────────
    ax_pc1 = fig.add_subplot(gs[1, 1])

    from scipy.stats import pearsonr
    all_pc1 = []; all_pos_rel = []; all_trope_plot = []

    for s in secuencias:
        emb_s  = scaler.transform(s["embeddings"])
        coords = pca2.transform(emb_s)
        n      = len(coords)
        pos    = np.linspace(0, 1, n)
        all_pc1.extend(coords[:, 0].tolist())
        all_pos_rel.extend(pos.tolist())
        all_trope_plot.extend([s["trope"]] * n)

    all_pc1      = np.array(all_pc1)
    all_pos_rel  = np.array(all_pos_rel)
    all_trope_plot = np.array(all_trope_plot)

    for t_idx in range(3):
        if tropo_filter is not None and t_idx != tropo_filter:
            continue
        mask = all_trope_plot == t_idx
        if mask.sum() < 5:
            continue
        pc1_t = all_pc1[mask]
        pos_t = all_pos_rel[mask]

        # Regresión: ¿PC1 cambia con posición?
        r, p = pearsonr(pos_t, pc1_t)

        # Scatter con transparencia
        ax_pc1.scatter(pos_t, pc1_t,
                       color=TROPE_COLORS[t_idx], alpha=0.15,
                       s=8, edgecolors="none")

        # Línea de tendencia
        m, b   = np.polyfit(pos_t, pc1_t, 1)
        x_line = np.linspace(0, 1, 50)
        ax_pc1.plot(x_line, m * x_line + b,
                    color=TROPE_COLORS[t_idx], linewidth=2.5,
                    label=f"{TROPE_NAMES[t_idx]} (r={r:.2f}, p={p:.3f})")

    ax_pc1.set_xlabel("Posición relativa en el fanfic (0=inicio, 1=fin)")
    ax_pc1.set_ylabel("PC1")
    ax_pc1.set_title("¿PC1 tiene tendencia temporal?\n"
                     "Correlación posición narrativa × PC1",
                     fontweight="bold")
    ax_pc1.legend(fontsize=8); ax_pc1.grid(alpha=0.3)
    ax_pc1.axhline(0, color="gray", linewidth=0.5, alpha=0.5)

    tropo_str = TROPE_NAMES.get(tropo_filter, "todos") \
                if tropo_filter is not None else "todos los tropos"
    plt.suptitle(f"Trayectorias emocionales en espacio PCA — {tropo_str}",
                 fontsize=13, fontweight="bold")

    os.makedirs(configl.FIGURES_DIR, exist_ok=True)
    if save:
        fname = f"13_pca_trayectorias_{tropo_str.replace(' ', '_')}.png"
        plt.savefig(f"{configl.FIGURES_DIR}{fname}",
                    dpi=150, bbox_inches="tight", facecolor="white")
    plt.show()
    print(f"✓ {len(muestra)} fanfics ploteados")


if __name__ == "__main__":
    # Todos los tropos
    pca_trayectorias()
    # Por tropo individual
    for t in range(3):
        pca_trayectorias(tropo_filter=t)
