"""
pca_lstm_ventanas.py — PCA de los hidden states de la WindowLSTM

A diferencia del PCA de embeddings RoBERTa (que no mostraba tendencia temporal),
aquí visualizamos lo que la LSTM aprendió a construir sobre esos embeddings:
los hidden states intermedios por ventana dentro de cada capítulo.

Pregunta central: ¿la LSTM organiza el espacio por posición narrativa?
Si sí → hay estructura temporal aprendida que RoBERTa no tenía.
Si no → la LSTM está capturando otro tipo de información.

Uso:
    from pca_lstm_ventanas import extraer_window_states, pca_lstm_trayectorias
    estados = extraer_window_states()
    pca_lstm_trayectorias(estados)
"""

import os
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import configl
from model import JointModel
from train import FanficDataset, create_dataloaders

TROPE_NAMES  = {0: "hurt_comfort", 1: "fluff", 2: "slow_burn"}
TROPE_COLORS = {0: "#E24B4A", 1: "#1D9E75", 2: "#7B68D8"}


def extraer_window_states(
    pkl_path:   str = "data/features/emotional_features.pkl",
    model_path: str = "models/best.pt",
    save_path:  str = "results/window_states.pkl",
    force:      bool = False,
) -> list:
    """
    Corre el forward pass con return_window_states=True.
    Guarda y retorna lista de dicts — uno por fanfic en el test set.

    Estructura de cada dict:
        title, fandom, trope_real, trope_pred, num_chapters,
        window_states: lista de arrays (n_wins, 256), uno por capítulo
                       — hidden states de la WindowLSTM por ventana
        window_counts: lista de ints — ventanas reales por capítulo
    """
    if not force and os.path.exists(save_path):
        print(f"Cargando desde {save_path}...")
        with open(save_path, "rb") as f:
            return pickle.load(f)

    print("Extrayendo hidden states de WindowLSTM...")
    with open(pkl_path, "rb") as f:
        features = pickle.load(f)

    ds = FanficDataset(features)
    _, _, test_dl = create_dataloaders(ds, batch_size=1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = JointModel()
    model.load_state_dict(torch.load(model_path, map_location=device,
                                     weights_only=True))
    model.to(device)
    model.eval()

    resultados = []
    with torch.no_grad():
        for i, batch in enumerate(test_dl):
            we  = batch["window_embs"].to(device)
            wc  = batch["window_counts"].to(device)
            ce  = batch["comment_embs"].to(device)
            cl  = batch["length"].to(device)

            out = model(we, wc, ce, cl,
                        return_window_states=True)

            n_caps     = int(cl[0].item())
            trope_real = int(batch["trope"][0].item())
            trope_pred = int(out["trope_logits"].argmax(1)[0].item())

            # window_states: lista de tensores (B, n_wins, 256) por capítulo
            # batch_size=1 → tomamos [0]
            cap_states = []
            cap_counts = []
            if "window_states" in out:
                for c_idx, states_tensor in enumerate(out["window_states"]):
                    if c_idx >= n_caps:
                        break
                    n_real = int(wc[0, c_idx].item())
                    # Recortar al número real de ventanas (sin padding)
                    states_np = states_tensor[0, :n_real, :].cpu().numpy()
                    cap_states.append(states_np)   # (n_wins_real, 256)
                    cap_counts.append(n_real)

            titulo = features[i % len(features)].get("title", f"fanfic_{i}")

            resultados.append({
                "title":        titulo,
                "fandom":       features[i % len(features)].get("fandom", ""),
                "trope_real":   trope_real,
                "trope_pred":   trope_pred,
                "num_chapters": n_caps,
                "window_states": cap_states,   # lista de (n_wins, 256)
                "window_counts": cap_counts,
            })

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(resultados, f)
    print(f"✓ {len(resultados)} fanfics → {save_path}")
    return resultados


def pca_lstm_trayectorias(
    estados:              list,
    tropo_filter:         int  = None,
    max_fanfics_por_tropo: int = 10,
    save:                 bool = True,
):
    """
    PCA de los hidden states de la WindowLSTM — 4 paneles:
      A) Varianza explicada acumulada (256-dim → cuántos PCs)
      B) Trayectorias por fanfic coloreadas por tropo
      C) Trayectorias coloreadas por posición intra-capítulo (gradiente)
      D) PC1 vs posición — ¿hay tendencia temporal que RoBERTa no tenía?
    """
    # Recopilar todos los hidden states con metadatos
    all_states, all_tropes, all_pos, all_cap_pos = [], [], [], []

    for d in estados:
        if tropo_filter is not None and d["trope_real"] != tropo_filter:
            continue
        n_caps = d["num_chapters"]
        for c_idx, states in enumerate(d["window_states"]):
            n_wins = states.shape[0]
            for w_idx in range(n_wins):
                all_states.append(states[w_idx])
                all_tropes.append(d["trope_real"])
                # Posición intra-capítulo (0=inicio cap, 1=fin cap)
                all_pos.append(w_idx / max(n_wins - 1, 1))
                # Posición inter-capítulo (0=primer cap, 1=último cap)
                all_cap_pos.append(c_idx / max(n_caps - 1, 1))

    if not all_states:
        print("No hay window_states disponibles.")
        print("Asegúrate de usar model.py actualizado (con _lstm_with_states).")
        return

    X        = np.stack(all_states)       # (N, 256)
    tropes   = np.array(all_tropes)
    pos      = np.array(all_pos)           # posición intra-capítulo
    cap_pos  = np.array(all_cap_pos)       # posición inter-capítulo

    print(f"PCA sobre {len(X)} hidden states de 256-dim (WindowLSTM)...")

    scaler  = StandardScaler()
    X_s     = scaler.fit_transform(X)

    # Varianza acumulada
    pca_full = PCA(n_components=min(50, X.shape[0], X.shape[1]),
                   random_state=42)
    pca_full.fit(X_s)
    var_cum = np.cumsum(pca_full.explained_variance_ratio_)

    pca2    = PCA(n_components=2, random_state=42)
    X_pca   = pca2.fit_transform(X_s)
    var2    = pca2.explained_variance_ratio_

    import random
    random.seed(42)
    fanfics_por_tropo = {}
    for d in estados:
        if tropo_filter is not None and d["trope_real"] != tropo_filter:
            continue
        if d["window_states"]:
            fanfics_por_tropo.setdefault(d["trope_real"], []).append(d)
    muestra = []
    for t, flist in fanfics_por_tropo.items():
        muestra.extend(random.sample(flist,
                       min(max_fanfics_por_tropo, len(flist))))

    fig = plt.figure(figsize=(20, 16))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # ── A) Varianza acumulada ─────────────────────────────────────────────────
    ax_var = fig.add_subplot(gs[0, 0])
    n_pcs  = len(var_cum)
    ax_var.plot(range(1, n_pcs + 1), var_cum, "-o",
                color="#7B68D8", linewidth=2, markersize=4)
    for thresh, color, label in [(0.5, "#F4A261", "50%"),
                                  (0.7, "#E24B4A", "70%"),
                                  (0.9, "#2E86AB", "90%")]:
        ax_var.axhline(thresh, color=color, linestyle="--",
                       alpha=0.7, label=f"{label} var")
        idx = np.searchsorted(var_cum, thresh)
        if idx < n_pcs:
            ax_var.annotate(f"{idx+1} PCs",
                            xy=(idx + 1, thresh),
                            xytext=(idx + 3, thresh - 0.04),
                            fontsize=8, color=color,
                            arrowprops=dict(arrowstyle="->",
                                            color=color, lw=0.8))
    ax_var.set_xlabel("Número de componentes")
    ax_var.set_ylabel("Varianza acumulada explicada")
    ax_var.set_title("Varianza acumulada — hidden states WindowLSTM (256-dim)\n"
                     "¿Más concentrada que RoBERTa (768-dim)?",
                     fontweight="bold")
    ax_var.legend(fontsize=8); ax_var.grid(alpha=0.3)

    # ── B) Trayectorias por fanfic coloreadas por tropo ───────────────────────
    ax_traj = fig.add_subplot(gs[0, 1])

    for d in muestra:
        color = TROPE_COLORS[d["trope_real"]]
        # Concatenar ventanas de todos los capítulos en orden
        if not d["window_states"]:
            continue
        all_wins = np.concatenate(d["window_states"], axis=0)
        all_wins_s = scaler.transform(all_wins)
        coords   = pca2.transform(all_wins_s)

        ax_traj.plot(coords[:, 0], coords[:, 1],
                     "-", color=color, alpha=0.35, linewidth=1.2)
        # Inicio
        ax_traj.scatter(coords[0, 0], coords[0, 1],
                        color=color, s=40, zorder=5,
                        edgecolors="white", linewidths=0.5)
        # Fin con flecha
        if len(coords) > 1:
            ax_traj.annotate("",
                xy=(coords[-1, 0], coords[-1, 1]),
                xytext=(coords[-2, 0], coords[-2, 1]),
                arrowprops=dict(arrowstyle="->", color=color,
                                lw=1.2, alpha=0.7))

    for t_idx in TROPE_NAMES:
        if tropo_filter is not None and t_idx != tropo_filter:
            continue
        ax_traj.plot([], [], "-o", color=TROPE_COLORS[t_idx],
                     label=TROPE_NAMES[t_idx], linewidth=1.5)
    ax_traj.legend(fontsize=8)
    ax_traj.set_xlabel(f"PC1 ({var2[0]*100:.1f}% var)")
    ax_traj.set_ylabel(f"PC2 ({var2[1]*100:.1f}% var)")
    ax_traj.set_title("Trayectorias WindowLSTM en espacio PCA\n"
                      "○=inicio → →=fin  (cada línea = un fanfic completo)",
                      fontweight="bold")
    ax_traj.grid(alpha=0.2)

    # ── C) Trayectorias con gradiente de posición intra-capítulo ─────────────
    ax_grad = fig.add_subplot(gs[1, 0])

    for d in muestra:
        if not d["window_states"]:
            continue
        # Por capítulo — trayectoria dentro de cada capítulo
        for c_idx, states in enumerate(d["window_states"]):
            if states.shape[0] < 2:
                continue
            states_s = scaler.transform(states)
            coords   = pca2.transform(states_s)
            n        = len(coords)
            pos_arr  = np.linspace(0, 1, n)

            points = coords.reshape(-1, 1, 2)
            segs   = np.concatenate([points[:-1], points[1:]], axis=1)
            lc     = LineCollection(segs, cmap="plasma",
                                    norm=plt.Normalize(0, 1),
                                    alpha=0.4, linewidth=1.5)
            lc.set_array(pos_arr[:-1])
            ax_grad.add_collection(lc)

    ax_grad.autoscale()
    sm = plt.cm.ScalarMappable(cmap="plasma", norm=plt.Normalize(0, 1))
    sm.set_array([])
    plt.colorbar(sm, ax=ax_grad,
                 label="Posición intra-capítulo (0=inicio, 1=fin)",
                 fraction=0.03)
    ax_grad.set_xlabel(f"PC1 ({var2[0]*100:.1f}% var)")
    ax_grad.set_ylabel(f"PC2 ({var2[1]*100:.1f}% var)")
    ax_grad.set_title("Trayectorias intra-capítulo (WindowLSTM)\n"
                      "¿Las ventanas de un capítulo tienen dirección en el PCA?",
                      fontweight="bold")
    ax_grad.grid(alpha=0.2)

    # ── D) PC1 vs posición — comparativa entre tropos ────────────────────────
    ax_pc1 = fig.add_subplot(gs[1, 1])

    for t_idx in range(3):
        if tropo_filter is not None and t_idx != tropo_filter:
            continue
        mask = tropes == t_idx
        if mask.sum() < 5:
            continue
        pc1_t = X_pca[mask, 0]
        pos_t = pos[mask]

        r, p = pearsonr(pos_t, pc1_t)

        ax_pc1.scatter(pos_t, pc1_t, color=TROPE_COLORS[t_idx],
                       alpha=0.12, s=8, edgecolors="none")
        m, b   = np.polyfit(pos_t, pc1_t, 1)
        x_line = np.linspace(0, 1, 50)
        ax_pc1.plot(x_line, m * x_line + b,
                    color=TROPE_COLORS[t_idx], linewidth=2.5,
                    label=f"{TROPE_NAMES[t_idx]}\nr={r:.3f}, p={p:.3f}")

    ax_pc1.set_xlabel("Posición intra-capítulo (0=inicio, 1=fin)")
    ax_pc1.set_ylabel("PC1 (WindowLSTM hidden states)")
    ax_pc1.set_title("¿PC1 tiene tendencia temporal en la WindowLSTM?\n"
                     "Comparar con RoBERTa (sin tendencia) — ¿diferencia?",
                     fontweight="bold")
    ax_pc1.legend(fontsize=8); ax_pc1.grid(alpha=0.3)
    ax_pc1.axhline(0, color="gray", linewidth=0.5, alpha=0.5)

    tropo_str = TROPE_NAMES.get(tropo_filter, "todos") \
                if tropo_filter is not None else "todos"
    plt.suptitle(f"Hidden states WindowLSTM — espacio PCA — {tropo_str}",
                 fontsize=13, fontweight="bold")

    os.makedirs(configl.FIGURES_DIR, exist_ok=True)
    if save:
        fname = f"14_pca_lstm_{tropo_str.replace(' ', '_')}.png"
        plt.savefig(f"{configl.FIGURES_DIR}{fname}",
                    dpi=150, bbox_inches="tight", facecolor="white")
    plt.show()
    print(f"  PC1 varianza: {var2[0]*100:.1f}%  |  "
          f"PC1+PC2: {sum(var2)*100:.1f}%")
