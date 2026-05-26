"""
═══════════════════════════════════════════════════════════════════
analysis.py — Análisis completo del modelo JointModel

Secciones:
  0. Extracción de activaciones (forward pass sobre test set)
  1. Exportar datos y resultados
  2. Desempeño general del modelo
  3. Arco emocional + comentarios + dependencias
  4. Análisis de atención (en qué se fija la red por tropo)
  5. PCA de hidden states
  6. Análisis adicionales recomendados

Uso desde notebook:
    from analysis import *
    datos = extraer_activaciones()
    analisis_desempeno(datos)
    visualizar_arcos_emocionales(datos)
    analizar_atencion(datos)
    pca_hidden_states(datos)
═══════════════════════════════════════════════════════════════════
"""

import os
import pickle
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, accuracy_score
)
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import cosine
import configl
from model import JointModel
from train import FanficDataset, create_dataloaders

# ── Configuración de estilo ───────────────────────────────────────────────────
TROPE_NAMES  = {0: "hurt_comfort", 1: "fluff", 2: "slow_burn"}
TROPE_COLORS = {0: "#E24B4A", 1: "#1D9E75", 2: "#7B68D8"}
TROPE_LABELS = list(TROPE_NAMES.values())

EMOTION_LABELS = configl.EMOTION_LABELS

# Emociones de alta varianza — las más discriminativas entre tropos
HIGH_VARIANCE_EMOTIONS = [
    "grief", "joy", "love", "sadness", "caring",
    "admiration", "fear", "relief", "excitement", "neutral"
]

os.makedirs(configl.RESULTS_DIR, exist_ok=True)
os.makedirs(configl.FIGURES_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# 0. EXTRACCIÓN DE ACTIVACIONES
# ══════════════════════════════════════════════════════════════════════════════
def extraer_activaciones(
    pkl_path: str = "data/features/emotional_features.pkl",
    model_path: str = "models/best.pt",
    save_path: str = "results/datos_visualizacion.pkl",
    force: bool = False,
) -> list:
    """
    Corre el forward pass sobre el test set con return_attention=True
    y return_hidden=True. Guarda y retorna una lista de dicts — uno por fanfic.

    Estructura de cada dict:
        title, fandom, trope_real, trope_pred, trope_probs,
        num_chapters,
        attention_weights:      (n_caps,)      ← pesos de ChapterAttention
        arc_hidden:             (n_caps, 256)  ← hidden states de ArcLSTM
        window_vecs:            (n_caps, 256)  ← salida WindowLSTM por capítulo
        reader_emotions_real:   (n_caps, 28)   ← ground truth GoEmotions
        reader_emotions_pred:   (n_caps, 28)   ← predicción del modelo
    """
    if not force and os.path.exists(save_path):
        print(f"Cargando datos existentes desde {save_path}...")
        with open(save_path, "rb") as f:
            return pickle.load(f)

    print("Cargando dataset y modelo...")
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

    # Mapa título → metadata original
    meta_map = {f["title"]: f for f in features}

    resultados = []
    with torch.no_grad():
        for i, batch in enumerate(test_dl):
            we  = batch["window_embs"].to(device)
            wc  = batch["window_counts"].to(device)
            ce  = batch["comment_embs"].to(device)
            cl  = batch["length"].to(device)

            out = model(we, wc, ce, cl,
                        return_attention=True, return_hidden=True)

            n_caps      = int(cl[0].item())
            trope_real  = int(batch["trope"][0].item())
            trope_pred  = int(out["trope_logits"].argmax(1)[0].item())
            trope_probs = torch.softmax(out["trope_logits"], dim=1)[0].cpu().numpy()

            attn    = out["attention_weights"][0].cpu().numpy()[:n_caps]
            arc_h   = out["hidden_states"][0].cpu().numpy()[:n_caps]
            win_v   = out["window_vecs"][0].cpu().numpy()[:n_caps]
            r_pred  = out["reader_pred"][0].cpu().numpy()[:n_caps]
            r_real  = batch["reader_arc"][0].numpy()[:n_caps]

            # Recuperar metadata — batch_size=1 así que tomamos el primer título
            # FanficDataset no guarda title directamente en el batch,
            # lo recuperamos por posición
            titulo = features[i % len(features)].get("title", f"fanfic_{i}")

            resultados.append({
                "title":                titulo,
                "fandom":               features[i % len(features)].get("fandom", ""),
                "trope_real":           trope_real,
                "trope_pred":           trope_pred,
                "trope_probs":          trope_probs,
                "num_chapters":         n_caps,
                "attention_weights":    attn,
                "arc_hidden":           arc_h,
                "window_vecs":          win_v,
                "reader_emotions_real": r_real,
                "reader_emotions_pred": r_pred,
            })

    with open(save_path, "wb") as f:
        pickle.dump(resultados, f)
    print(f"✓ {len(resultados)} fanfics exportados → {save_path}")
    return resultados


# ══════════════════════════════════════════════════════════════════════════════
# 1. EXPORTAR DATOS Y RESULTADOS
# ══════════════════════════════════════════════════════════════════════════════
def exportar_resultados(datos: list, history: dict = None):
    """
    Exporta resultados en formatos usables:
      - results/test_predictions.csv   ← predicciones por fanfic
      - results/emotion_predictions.csv ← predicciones emocionales por capítulo
      - results/summary.json            ← métricas globales
    """
    # CSV de predicciones de tropo
    rows_trope = []
    rows_emo   = []

    for d in datos:
        rows_trope.append({
            "title":        d["title"],
            "fandom":       d["fandom"],
            "trope_real":   TROPE_NAMES[d["trope_real"]],
            "trope_pred":   TROPE_NAMES[d["trope_pred"]],
            "correct":      d["trope_real"] == d["trope_pred"],
            "num_chapters": d["num_chapters"],
            "prob_hurt_comfort": d["trope_probs"][0],
            "prob_fluff":        d["trope_probs"][1],
            "prob_slow_burn":    d["trope_probs"][2],
        })
        for c in range(d["num_chapters"]):
            row = {
                "title":   d["title"],
                "trope":   TROPE_NAMES[d["trope_real"]],
                "chapter": c + 1,
                "attn_weight": d["attention_weights"][c],
            }
            for e_idx, e_name in enumerate(EMOTION_LABELS):
                row[f"real_{e_name}"]  = d["reader_emotions_real"][c, e_idx]
                row[f"pred_{e_name}"]  = d["reader_emotions_pred"][c, e_idx]
            rows_emo.append(row)

    df_trope = pd.DataFrame(rows_trope)
    df_emo   = pd.DataFrame(rows_emo)

    df_trope.to_csv(f"{configl.RESULTS_DIR}test_predictions.csv", index=False)
    df_emo.to_csv(f"{configl.RESULTS_DIR}emotion_predictions.csv", index=False)

    # Métricas globales
    y_true = [d["trope_real"] for d in datos]
    y_pred = [d["trope_pred"] for d in datos]
    summary = {
        "accuracy":  accuracy_score(y_true, y_pred),
        "f1_macro":  f1_score(y_true, y_pred, average="macro"),
        "n_test":    len(datos),
        "baseline":  1 / configl.NUM_TROPES,
        "per_trope": classification_report(y_true, y_pred,
                         target_names=TROPE_LABELS, output_dict=True),
    }
    if history:
        summary["best_epoch"] = int(np.argmin(history.get("val_loss", [0]))) + 1
        summary["best_val_loss"] = float(min(history.get("val_loss", [0])))

    with open(f"{configl.RESULTS_DIR}summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"✓ Exportado: test_predictions.csv, emotion_predictions.csv, summary.json")
    return df_trope, df_emo


# ══════════════════════════════════════════════════════════════════════════════
# 2. DESEMPEÑO GENERAL DEL MODELO
# ══════════════════════════════════════════════════════════════════════════════
def analisis_desempeno(datos: list, history: dict = None):
    """
    Genera figura de 4 paneles:
      A) Curvas de pérdida train/val
      B) Curvas de accuracy train/val
      C) Matriz de confusión
      D) F1 por tropo con línea de baseline
    """
    y_true = [d["trope_real"] for d in datos]
    y_pred = [d["trope_pred"] for d in datos]

    fig = plt.figure(figsize=(18, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # ── A) Curvas de pérdida ──
    if history:
        ax = fig.add_subplot(gs[0, 0])
        epochs = range(1, len(history["train_loss"]) + 1)
        ax.plot(epochs, history["train_loss"], label="Train", color="#2E86AB")
        ax.plot(epochs, history["val_loss"],   label="Val",   color="#E84855",
                linestyle="--")
        ax.set_title("Pérdida total", fontweight="bold")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.legend(); ax.grid(alpha=0.3)

        # ── B) Accuracy ──
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(epochs, history["train_accuracy"], label="Train", color="#2E86AB")
        ax2.plot(epochs, history["val_accuracy"],   label="Val",   color="#E84855",
                 linestyle="--")
        ax2.axhline(1/3, color="gray", linestyle=":", label=f"Baseline ({1/3:.2f})")
        ax2.set_title("Accuracy — tropo", fontweight="bold")
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
        ax2.legend(); ax2.grid(alpha=0.3)

        # ── B2) Loss reader ──
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(epochs, history["train_reader_loss"], label="Train", color="#7B68D8")
        ax3.plot(epochs, history["val_reader_loss"],   label="Val",   color="#F4A261",
                 linestyle="--")
        ax3.set_title("Pérdida tarea 2 (MSE emocional)", fontweight="bold")
        ax3.set_xlabel("Epoch"); ax3.set_ylabel("MSE")
        ax3.legend(); ax3.grid(alpha=0.3)

    # ── C) Matriz de confusión ──
    ax_cm = fig.add_subplot(gs[1, 0])
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm_norm, annot=cm, fmt="d", cmap="Blues",
                xticklabels=TROPE_LABELS, yticklabels=TROPE_LABELS,
                ax=ax_cm, linewidths=0.5)
    ax_cm.set_title("Matriz de confusión", fontweight="bold")
    ax_cm.set_xlabel("Predicho"); ax_cm.set_ylabel("Real")
    ax_cm.tick_params(axis="x", rotation=30)

    # ── D) F1 por tropo ──
    ax_f1 = fig.add_subplot(gs[1, 1])
    report = classification_report(y_true, y_pred, target_names=TROPE_LABELS,
                                   output_dict=True)
    f1s    = [report[t]["f1-score"] for t in TROPE_LABELS]
    bars   = ax_f1.bar(TROPE_LABELS, f1s,
                       color=[TROPE_COLORS[i] for i in range(3)],
                       alpha=0.85, edgecolor="white")
    ax_f1.axhline(1/3, color="gray", linestyle="--", label=f"Baseline ({1/3:.2f})")
    ax_f1.set_ylim(0, 1)
    ax_f1.set_title("F1-score por tropo", fontweight="bold")
    ax_f1.set_ylabel("F1"); ax_f1.legend(); ax_f1.grid(alpha=0.3, axis="y")
    for bar, val in zip(bars, f1s):
        ax_f1.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                   f"{val:.2f}", ha="center", fontsize=10)

    # ── E) Accuracy por longitud de fanfic ──
    ax_len = fig.add_subplot(gs[1, 2])
    df = pd.DataFrame({
        "n_caps":    [d["num_chapters"] for d in datos],
        "correct":   [d["trope_real"] == d["trope_pred"] for d in datos],
    })
    df["bucket"] = df["n_caps"].apply(
        lambda x: f"{((x-1)//3)*3+1}-{((x-1)//3)*3+3}"
    )
    acc_by_len = df.groupby("bucket")["correct"].mean()
    ax_len.bar(acc_by_len.index, acc_by_len.values, color="#2E86AB", alpha=0.8)
    ax_len.axhline(1/3, color="gray", linestyle="--")
    ax_len.set_title("Accuracy por nº capítulos", fontweight="bold")
    ax_len.set_xlabel("Rango de capítulos"); ax_len.set_ylabel("Accuracy")
    ax_len.grid(alpha=0.3, axis="y")

    plt.suptitle("Análisis de Desempeño del Modelo", fontsize=14, fontweight="bold")
    plt.savefig(f"{configl.FIGURES_DIR}01_desempeno.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\nAccuracy test: {accuracy_score(y_true, y_pred):.3f}")
    print(f"F1-macro test: {f1_score(y_true, y_pred, average='macro'):.3f}")
    print(f"Baseline:      {1/3:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. ARCOS EMOCIONALES + COMENTARIOS + DEPENDENCIAS
# ══════════════════════════════════════════════════════════════════════════════
def visualizar_arcos_emocionales(datos: list, n_ejemplos: int = 2):
    """
    Por cada tropo, muestra n_ejemplos fanfics con:
      - Arco emocional real vs predicho (emociones clave)
      - Peso de atención por capítulo
      - Correlación capítulo-emoción (mapa de calor)
    """
    for tropo_idx in range(configl.NUM_TROPES):
        tropo_nombre = TROPE_NAMES[tropo_idx]
        ejemplos = [d for d in datos if d["trope_real"] == tropo_idx][:n_ejemplos]

        for d in ejemplos:
            n = d["num_chapters"]
            caps = np.arange(1, n + 1)
            r_real = d["reader_emotions_real"]   # (n, 28)
            r_pred = d["reader_emotions_pred"]   # (n, 28)
            attn   = d["attention_weights"]      # (n,)

            # Emociones a mostrar: las de mayor varianza en este fanfic
            var_local = r_real.var(axis=0)
            top_emos  = np.argsort(var_local)[::-1][:6]
            emo_names = [EMOTION_LABELS[i] for i in top_emos]

            fig, axes = plt.subplots(3, 1, figsize=(14, 12))
            color = TROPE_COLORS[tropo_idx]

            # ── Panel 1: arco emocional real vs predicho ──
            ax1 = axes[0]
            for i, (emo_idx, emo_name) in enumerate(zip(top_emos, emo_names)):
                alpha = 1.0 if i < 3 else 0.4
                ax1.plot(caps, r_real[:, emo_idx], "-o",
                         label=f"{emo_name} (real)", alpha=alpha,
                         linewidth=2)
                ax1.plot(caps, r_pred[:, emo_idx], "--x",
                         label=f"{emo_name} (pred)", alpha=alpha * 0.7,
                         linewidth=1.5)
            ax1.set_title(f"Arco emocional — {tropo_nombre}\n«{d['title'][:60]}»",
                         fontweight="bold")
            ax1.set_xlabel("Capítulo"); ax1.set_ylabel("Intensidad emocional")
            ax1.legend(loc="upper right", ncol=2, fontsize=8)
            ax1.grid(alpha=0.3)

            # ── Panel 2: pesos de atención ──
            ax2 = axes[1]
            ax2.bar(caps, attn, color=color, alpha=0.8, edgecolor="white")
            ax2.set_title("Atención de la red por capítulo\n"
                          "(qué capítulos influyen más en la predicción de tropo)",
                          fontweight="bold")
            ax2.set_xlabel("Capítulo"); ax2.set_ylabel("Peso de atención")
            ax2.grid(alpha=0.3, axis="y")

            # ── Panel 3: correlación emoción-capítulo (dependencias) ──
            ax3 = axes[2]
            # Mapa de calor: capítulos × emociones top
            emo_matrix = r_real[:, top_emos].T   # (6, n_caps)
            im = ax3.imshow(emo_matrix, aspect="auto", cmap="RdYlGn",
                            vmin=0, vmax=0.5)
            ax3.set_xticks(range(n)); ax3.set_xticklabels(caps)
            ax3.set_yticks(range(len(emo_names)))
            ax3.set_yticklabels(emo_names)
            ax3.set_title("Distribución emocional de comentarios por capítulo\n"
                          "(dependencias temporales)", fontweight="bold")
            ax3.set_xlabel("Capítulo")
            plt.colorbar(im, ax=ax3, fraction=0.03)

            plt.tight_layout()
            fname = f"02_arco_{tropo_nombre}_{d['title'][:20].replace(' ', '_')}.png"
            plt.savefig(f"{configl.FIGURES_DIR}{fname}", dpi=150, bbox_inches="tight")
            plt.show()


def analizar_dependencias_tropo(datos: list):
    """
    Para cada tropo: correlación de Pearson entre posición del capítulo
    y cada emoción — muestra qué emociones tienen tendencia temporal.

    Por ejemplo: en hurt_comfort, ¿grief decrece y relief aumenta con el tiempo?
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 7), sharey=True)

    for t_idx, ax in enumerate(axes):
        tropo_nombre = TROPE_NAMES[t_idx]
        ejemplos = [d for d in datos if d["trope_real"] == t_idx]

        # Acumular correlaciones posición-emoción
        all_corrs = []
        for d in ejemplos:
            n = d["num_chapters"]
            if n < 3:
                continue
            pos = np.arange(n) / max(n - 1, 1)   # posición normalizada 0-1
            for e_idx in range(28):
                emo_seq = d["reader_emotions_real"][:, e_idx]
                if emo_seq.std() > 1e-6:
                    r, _ = pearsonr(pos, emo_seq)
                    all_corrs.append({"emotion": EMOTION_LABELS[e_idx], "r": r})

        if not all_corrs:
            continue

        df_corr = pd.DataFrame(all_corrs).groupby("emotion")["r"].mean()
        df_corr = df_corr.sort_values()

        # Mostrar solo las 8 con mayor |correlación|
        df_top = df_corr.reindex(df_corr.abs().nlargest(8).index).sort_values()
        colors = ["#E24B4A" if v < 0 else "#1D9E75" for v in df_top.values]

        ax.barh(df_top.index, df_top.values, color=colors, alpha=0.85,
                edgecolor="white")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(f"{tropo_nombre}\n(rojo=decrece, verde=aumenta)",
                     fontweight="bold")
        ax.set_xlabel("Correlación de Pearson promedio")
        ax.grid(alpha=0.3, axis="x")

    plt.suptitle("Dependencias temporales: ¿qué emociones cambian a lo largo del fanfic?",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{configl.FIGURES_DIR}03_dependencias_temporales.png",
                dpi=150, bbox_inches="tight")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# 4. ANÁLISIS DE ATENCIÓN — EN QUÉ SE FIJA LA RED POR TROPO
# ══════════════════════════════════════════════════════════════════════════════
def analizar_atencion(datos: list):
    """
    Tres visualizaciones de atención:
      A) Distribución de atención por posición relativa (inicio/medio/fin)
         por tropo — ¿la red se fija más en el clímax? ¿en la resolución?
      B) Correlación atención × emoción — ¿qué emociones del capítulo
         hacen que la red le preste más atención?
      C) Heatmap atención promedio por tropo y posición normalizada
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # ── A) Atención por posición relativa ──
    ax = axes[0]
    n_bins = 5
    bin_labels = [f"{int(i*100/n_bins)}-{int((i+1)*100/n_bins)}%" for i in range(n_bins)]
    for t_idx in range(3):
        ejemplos = [d for d in datos if d["trope_real"] == t_idx]
        bin_attn = np.zeros(n_bins)
        count    = 0
        for d in ejemplos:
            n    = d["num_chapters"]
            attn = d["attention_weights"]
            if n < 2:
                continue
            for cap_i, a in enumerate(attn):
                pos_rel = cap_i / (n - 1)
                b = min(int(pos_rel * n_bins), n_bins - 1)
                bin_attn[b] += a
            count += 1
        if count > 0:
            bin_attn /= count
        ax.plot(bin_labels, bin_attn, "-o",
                label=TROPE_NAMES[t_idx],
                color=TROPE_COLORS[t_idx], linewidth=2)

    ax.set_title("Distribución de atención por posición narrativa",
                 fontweight="bold")
    ax.set_xlabel("Posición en el fanfic (inicio → fin)")
    ax.set_ylabel("Peso de atención promedio")
    ax.legend(); ax.grid(alpha=0.3)

    # ── B) Correlación atención × emoción ──
    ax2 = axes[1]
    corr_by_tropo = {}
    for t_idx in range(3):
        ejemplos = [d for d in datos if d["trope_real"] == t_idx]
        emo_corrs = np.zeros(28)
        valid = 0
        for d in ejemplos:
            n = d["num_chapters"]
            if n < 3:
                continue
            attn = d["attention_weights"]
            for e_idx in range(28):
                emo_seq = d["reader_emotions_real"][:, e_idx]
                if emo_seq.std() > 1e-6 and attn.std() > 1e-6:
                    r, _ = pearsonr(attn, emo_seq)
                    emo_corrs[e_idx] += r
            valid += 1
        if valid > 0:
            emo_corrs /= valid
        corr_by_tropo[t_idx] = emo_corrs

    # Mostrar top emociones con mayor variación entre tropos
    corr_matrix = np.stack([corr_by_tropo[i] for i in range(3)])  # (3, 28)
    variability = corr_matrix.var(axis=0)
    top_emos    = np.argsort(variability)[::-1][:10]
    top_names   = [EMOTION_LABELS[i] for i in top_emos]

    data_plot = corr_matrix[:, top_emos]
    x = np.arange(len(top_names))
    width = 0.25
    for t_idx in range(3):
        ax2.bar(x + t_idx * width, data_plot[t_idx], width,
                label=TROPE_NAMES[t_idx], color=TROPE_COLORS[t_idx], alpha=0.85)
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(top_names, rotation=40, ha="right", fontsize=8)
    ax2.axhline(0, color="black", linewidth=0.7)
    ax2.set_title("Correlación atención × emoción por tropo\n"
                  "(qué emociones atraen la atención de la red)",
                  fontweight="bold")
    ax2.set_ylabel("r de Pearson promedio")
    ax2.legend(); ax2.grid(alpha=0.3, axis="y")

    # ── C) Heatmap de atención promedio ──
    ax3 = axes[2]
    n_pos = 10  # posiciones normalizadas
    heatmap_data = np.zeros((3, n_pos))
    for t_idx in range(3):
        ejemplos = [d for d in datos if d["trope_real"] == t_idx]
        counts   = np.zeros(n_pos)
        for d in ejemplos:
            n = d["num_chapters"]
            if n < 2:
                continue
            attn = d["attention_weights"]
            for cap_i, a in enumerate(attn):
                pos_rel = cap_i / (n - 1)
                b = min(int(pos_rel * n_pos), n_pos - 1)
                heatmap_data[t_idx, b] += a
                counts[b] += 1
        heatmap_data[t_idx] /= np.maximum(counts, 1)

    sns.heatmap(heatmap_data, ax=ax3, cmap="YlOrRd",
                xticklabels=[f"{int(i*10)}-{int((i+1)*10)}%" for i in range(n_pos)],
                yticklabels=TROPE_LABELS, linewidths=0.5)
    ax3.set_title("Mapa de atención por tropo y posición narrativa",
                  fontweight="bold")
    ax3.set_xlabel("Posición en el fanfic")
    ax3.tick_params(axis="x", rotation=40)

    plt.suptitle("Análisis de Atención — ¿En qué se fija la red para clasificar tropos?",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{configl.FIGURES_DIR}04_atencion.png", dpi=150, bbox_inches="tight")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# 5. PCA DE HIDDEN STATES
# ══════════════════════════════════════════════════════════════════════════════
def pca_hidden_states(datos: list):
    """
    Tres análisis de PCA:
      A) PCA de arc_hidden coloreado por tropo
         ¿Se separan los tropos en el espacio de representación?
      B) PCA coloreado por posición narrativa (inicio/medio/fin)
         ¿La LSTM organiza el espacio por posición en la obra?
      C) t-SNE como complemento al PCA
    """
    # Recopilar todos los hidden states con metadatos
    all_hidden = []
    all_tropes = []
    all_pos    = []
    all_fanfic = []

    for d in datos:
        n    = d["num_chapters"]
        h    = d["arc_hidden"]   # (n, 256)
        for cap_i in range(n):
            all_hidden.append(h[cap_i])
            all_tropes.append(d["trope_real"])
            all_pos.append(cap_i / max(n - 1, 1))   # 0=inicio, 1=fin
            all_fanfic.append(d["title"][:20])

    X      = np.stack(all_hidden)   # (N, 256)
    tropes = np.array(all_tropes)
    pos    = np.array(all_pos)

    # ── PCA ──
    pca    = PCA(n_components=2, random_state=42)
    X_pca  = pca.fit_transform(X)
    var_explained = pca.explained_variance_ratio_

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))

    # ── A) PCA por tropo ──
    ax = axes[0]
    for t_idx in range(3):
        mask = tropes == t_idx
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=TROPE_COLORS[t_idx], label=TROPE_NAMES[t_idx],
                   alpha=0.6, s=30, edgecolors="none")
    ax.set_xlabel(f"PC1 ({var_explained[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({var_explained[1]*100:.1f}% var)")
    ax.set_title("PCA de hidden states\ncoloreado por tropo", fontweight="bold")
    ax.legend(); ax.grid(alpha=0.2)

    # ── B) PCA por posición narrativa ──
    ax2 = axes[1]
    sc = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=pos,
                     cmap="plasma", alpha=0.6, s=30, edgecolors="none")
    plt.colorbar(sc, ax=ax2, label="Posición relativa (0=inicio, 1=fin)")
    ax2.set_xlabel(f"PC1 ({var_explained[0]*100:.1f}% var)")
    ax2.set_ylabel(f"PC2 ({var_explained[1]*100:.1f}% var)")
    ax2.set_title("PCA de hidden states\ncoloreado por posición narrativa",
                  fontweight="bold")
    ax2.grid(alpha=0.2)

    # ── C) t-SNE ──
    ax3 = axes[2]
    perplexity = min(30, len(X) // 4)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity,
            max_iter=1000)
    X_tsne = tsne.fit_transform(X)
    for t_idx in range(3):
        mask = tropes == t_idx
        ax3.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                    c=TROPE_COLORS[t_idx], label=TROPE_NAMES[t_idx],
                    alpha=0.6, s=30, edgecolors="none")
    ax3.set_title("t-SNE de hidden states\ncoloreado por tropo", fontweight="bold")
    ax3.legend(); ax3.grid(alpha=0.2)
    ax3.set_xlabel("t-SNE 1"); ax3.set_ylabel("t-SNE 2")

    plt.suptitle("Análisis del espacio de representación interna de la red",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{configl.FIGURES_DIR}05_pca_hidden.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Varianza explicada por PC
    pca_full = PCA(n_components=min(20, X.shape[1]), random_state=42)
    pca_full.fit(X)
    fig2, ax_var = plt.subplots(figsize=(8, 4))
    ax_var.bar(range(1, len(pca_full.explained_variance_ratio_) + 1),
               np.cumsum(pca_full.explained_variance_ratio_),
               color="#2E86AB", alpha=0.8)
    ax_var.axhline(0.9, color="red", linestyle="--", label="90% varianza")
    ax_var.set_xlabel("Componente principal")
    ax_var.set_ylabel("Varianza acumulada explicada")
    ax_var.set_title("Varianza explicada por los PCs", fontweight="bold")
    ax_var.legend(); ax_var.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{configl.FIGURES_DIR}05b_pca_varianza.png", dpi=150,
                bbox_inches="tight")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# 6. ANÁLISIS ADICIONALES RECOMENDADOS
# ══════════════════════════════════════════════════════════════════════════════
def perfil_emocional_por_tropo(datos: list):
    """
    RECOMENDADO A — Perfil emocional promedio por tropo.
    Heatmap: tropos × emociones — muestra la firma emocional de cada tropo.
    Conecta directamente con la hipótesis de ToM: cada tropo tiene un
    script emocional distinto que la red aprendió a reconocer.
    """
    fig, axes = plt.subplots(1, 2, figsize=(20, 7))

    # Real
    for ax_idx, (source, title) in enumerate([
        ("reader_emotions_real", "Distribución emocional REAL de comentarios"),
        ("reader_emotions_pred", "Distribución emocional PREDICHA por el modelo"),
    ]):
        ax = axes[ax_idx]
        matrix = np.zeros((3, 28))
        for t_idx in range(3):
            ejemplos = [d for d in datos if d["trope_real"] == t_idx]
            all_emo  = np.concatenate([d[source] for d in ejemplos], axis=0)
            matrix[t_idx] = all_emo.mean(axis=0)

        sns.heatmap(matrix, ax=ax, cmap="YlOrRd",
                    xticklabels=EMOTION_LABELS,
                    yticklabels=TROPE_LABELS,
                    linewidths=0.3, annot=False,vmin=0, vmax=0.15)
        ax.set_title(title, fontweight="bold")
        ax.tick_params(axis="x", rotation=60, labelsize=7)

    plt.suptitle("Perfil emocional por tropo — firma afectiva aprendida",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{configl.FIGURES_DIR}06_perfil_emocional_tropo.png",
                dpi=150, bbox_inches="tight")
    #plt.clim(0, 0.15)
    plt.show()


def similitud_arco_respuesta(datos: list):
    """
    RECOMENDADO B — Similitud coseno entre arco emocional del capítulo
    y distribución emocional de sus comentarios.

    Pregunta: ¿el arco emocional que construyó el autor es similar
    a la emoción que expresaron los lectores?
    → Si sí: evidencia de contagio afectivo.
    → Si varía por tropo: evidencia de que algunos tropos producen
      respuestas más congruentes que otros.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    data_by_tropo = {t: [] for t in range(3)}

    for d in datos:
        n = d["num_chapters"]
        for c in range(n):
            win_v   = d["window_vecs"][c]        # (256,) — arco del capítulo
            r_real  = d["reader_emotions_real"][c]  # (28,) — emoción lectores

            # Proyectar ambos a espacio comparable — usar PCA o simplemente
            # los primeros 28 dims del vector de capítulo como proxy
            # Aquí usamos norma coseno entre distribuciones emocionales
            # real vs predicha como proxy de congruencia
            r_pred  = d["reader_emotions_pred"][c]
            if r_real.sum() > 0 and r_pred.sum() > 0:
                sim = 1 - cosine(r_real, r_pred)
                data_by_tropo[d["trope_real"]].append(sim)

    positions = [1, 2, 3]
    bp_data   = [data_by_tropo[i] for i in range(3)]
    bp = ax.boxplot(bp_data, positions=positions, patch_artist=True,
                    widths=0.5, showfliers=False)
    for patch, t_idx in zip(bp["boxes"], range(3)):
        patch.set_facecolor(TROPE_COLORS[t_idx])
        patch.set_alpha(0.7)

    ax.set_xticks(positions)
    ax.set_xticklabels(TROPE_LABELS)
    ax.set_ylabel("Similitud coseno (real vs predicho)")
    ax.set_title("Congruencia emocional arco-respuesta por tropo\n"
                 "(similitud entre emoción predicha y emoción real de lectores)",
                 fontweight="bold")
    ax.grid(alpha=0.3, axis="y")

    # Añadir medias
    for i, vals in enumerate(bp_data):
        if vals:
            ax.text(i + 1, np.mean(vals) + 0.01, f"μ={np.mean(vals):.2f}",
                    ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{configl.FIGURES_DIR}07_similitud_arco_respuesta.png",
                dpi=150, bbox_inches="tight")
    plt.show()


def analisis_errores(datos: list):
    """
    RECOMENDADO C — Análisis de errores de clasificación.
    ¿Qué fanfics confunde el modelo y por qué?
    ¿Los errores tienen algún patrón emocional?
    """
    errores = [d for d in datos if d["trope_real"] != d["trope_pred"]]
    correctos = [d for d in datos if d["trope_real"] == d["trope_pred"]]

    print(f"\nErrores de clasificación: {len(errores)}/{len(datos)}")
    print(f"Accuracy: {len(correctos)/len(datos):.3f}\n")

    # Confianza del modelo en errores vs aciertos
    conf_errores  = [d["trope_probs"].max() for d in errores]
    conf_correctos = [d["trope_probs"].max() for d in correctos]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.hist(conf_correctos, bins=15, alpha=0.7, color="#1D9E75",
            label=f"Correctos (n={len(correctos)})", edgecolor="white")
    ax.hist(conf_errores, bins=15, alpha=0.7, color="#E24B4A",
            label=f"Errores (n={len(errores)})", edgecolor="white")
    ax.set_xlabel("Confianza máxima del modelo (softmax)")
    ax.set_ylabel("Frecuencia")
    ax.set_title("Confianza del modelo en aciertos vs errores",
                 fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3)

    # Longitud de fanfic en errores vs aciertos
    ax2 = axes[1]
    len_errores  = [d["num_chapters"] for d in errores]
    len_correctos = [d["num_chapters"] for d in correctos]
    ax2.boxplot([len_correctos, len_errores], labels=["Correctos", "Errores"],
                patch_artist=True,
                boxprops=dict(facecolor="#2E86AB", alpha=0.7))
    ax2.set_ylabel("Número de capítulos")
    ax2.set_title("Longitud del fanfic en aciertos vs errores",
                  fontweight="bold")
    ax2.grid(alpha=0.3, axis="y")

    plt.suptitle("Análisis de Errores de Clasificación", fontsize=13,
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{configl.FIGURES_DIR}08_analisis_errores.png",
                dpi=150, bbox_inches="tight")
    plt.show()

    # Tabla de errores
    print("\nFanfics mal clasificados:")
    print(f"{'Título':<40} {'Real':<15} {'Predicho':<15} {'Confianza'}")
    print("-" * 80)
    for d in errores:
        print(f"{d['title'][:38]:<40} "
              f"{TROPE_NAMES[d['trope_real']]:<15} "
              f"{TROPE_NAMES[d['trope_pred']]:<15} "
              f"{d['trope_probs'].max():.2f}")


# ══════════════════════════════════════════════════════════════════════════════
# FUNCIÓN PRINCIPAL — ejecuta todo en orden
# ══════════════════════════════════════════════════════════════════════════════
def ejecutar_analisis_completo(history: dict = None):
    """
    Ejecuta todo el pipeline de análisis en orden.
    Pasar history=history desde el entrenamiento para ver curvas de pérdida.
    """
    print("═" * 60)
    print("ANÁLISIS COMPLETO DEL MODELO")
    print("═" * 60)

    # 0. Extraer activaciones
    datos = extraer_activaciones()

    # 1. Exportar
    print("\n[1/7] Exportando resultados...")
    exportar_resultados(datos, history)

    # 2. Desempeño general
    print("\n[2/7] Análisis de desempeño...")
    analisis_desempeno(datos, history)

    # 3. Arcos emocionales
    print("\n[3/7] Visualizando arcos emocionales...")
    visualizar_arcos_emocionales(datos, n_ejemplos=1)

    # 4. Dependencias temporales
    print("\n[4/7] Analizando dependencias temporales...")
    analizar_dependencias_tropo(datos)

    # 5. Atención
    print("\n[5/7] Analizando atención de la red...")
    analizar_atencion(datos)

    # 6. PCA
    print("\n[6/7] PCA de hidden states...")
    pca_hidden_states(datos)

    # 7. Análisis adicionales
    print("\n[7/7] Análisis adicionales...")
    perfil_emocional_por_tropo(datos)
    similitud_arco_respuesta(datos)
    analisis_errores(datos)

    print(f"\n✓ Figuras guardadas en {configl.FIGURES_DIR}")
    print(f"✓ Datos exportados en {configl.RESULTS_DIR}")
    return datos


if __name__ == "__main__":
    ejecutar_analisis_completo()
