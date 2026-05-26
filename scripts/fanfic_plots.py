"""
fanfic_plots.py — Subplot por fanfic: arco emocional + atención + distribución

Uso:
    from fanfic_plots import guardar_subplots_fanfics
    guardar_subplots_fanfics(datos)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import configl

# ── Grupos de emociones (reducción 28 → 6) ───────────────────────────────────
EMOTION_GROUPS = {
    "afecto_positivo": ["admiration", "love", "gratitude", "caring"],
    "activación":      ["excitement", "nervousness", "desire", "curiosity"],
    "angustia":        ["grief", "sadness", "fear", "remorse"],
    "resolución":      ["relief", "joy", "optimism", "pride"],
    "rechazo":         ["disgust", "disapproval", "anger", "annoyance"],
    "ambiguo":         ["confusion", "realization", "surprise", "neutral"],
}

GROUP_COLORS = {
    "afecto_positivo": "#E84393",
    "activación":      "#F4A261",
    "angustia":        "#E24B4A",
    "resolución":      "#1D9E75",
    "rechazo":         "#7B3F00",
    "ambiguo":         "#7B68D8",
}

TROPE_COLORS = {0: "#E24B4A", 1: "#1D9E75", 2: "#7B68D8"}
TROPE_NAMES  = {0: "hurt_comfort", 1: "fluff", 2: "slow_burn"}

EMOTION_LABELS = configl.EMOTION_LABELS


def _colapsar_emociones(emo_matrix: np.ndarray) -> dict:
    """
    Colapsa (n_caps, 28) en (n_caps, 6) sumando por grupo.
    Retorna dict {group_name: array(n_caps,)}
    """
    result = {}
    for group, emotions in EMOTION_GROUPS.items():
        indices = [EMOTION_LABELS.index(e) for e in emotions
                   if e in EMOTION_LABELS]
        result[group] = emo_matrix[:, indices].sum(axis=1)
    return result


def plot_fanfic(d: dict, ax_arc, ax_attn, ax_dist, show_pred: bool = True):
    """
    Dibuja tres paneles para un fanfic:
      ax_arc:  arco emocional real (+ predicho si show_pred=True)
      ax_attn: pesos de atención por capítulo
      ax_dist: distribución emocional agrupada por capítulo (stacked bar)

    Parámetros:
        d:         dict de datos del fanfic (salida de extraer_activaciones)
        ax_arc:    eje matplotlib para el arco emocional
        ax_attn:   eje matplotlib para la atención
        ax_dist:   eje matplotlib para la distribución
        show_pred: si True, superpone la predicción del modelo en el arco
    """
    n    = d["num_chapters"]
    caps = np.arange(1, n + 1)
    color = TROPE_COLORS[d["trope_real"]]

    real_grouped = _colapsar_emociones(d["reader_emotions_real"])
    pred_grouped = _colapsar_emociones(d["reader_emotions_pred"]) if show_pred else None

    # ── Panel 1: arco emocional (grupos) ──────────────────────────────────────
    for group, values in real_grouped.items():
        ax_arc.plot(caps, values, "-o", color=GROUP_COLORS[group],
                    label=group, linewidth=2, markersize=4, alpha=0.9)
        if show_pred and pred_grouped:
            ax_arc.plot(caps, pred_grouped[group], "--",
                        color=GROUP_COLORS[group], linewidth=1.2,
                        alpha=0.5)

    ax_arc.set_ylabel("Intensidad emocional", fontsize=8)
    ax_arc.set_xlim(0.5, n + 0.5)
    ax_arc.set_xticks(caps)
    ax_arc.grid(alpha=0.25)
    ax_arc.tick_params(labelsize=7)

    # Línea discontinua = predicción
    if show_pred:
        ax_arc.plot([], [], "--", color="gray", alpha=0.5,
                    label="predicho", linewidth=1)

    ax_arc.legend(loc="upper right", fontsize=6, ncol=2,
                  framealpha=0.7)

    # ── Panel 2: pesos de atención ────────────────────────────────────────────
    bars = ax_attn.bar(caps, d["attention_weights"],
                       color=color, alpha=0.75, edgecolor="white",
                       linewidth=0.5)

    # Resaltar el capítulo con mayor atención
    max_attn_idx = int(d["attention_weights"].argmax())
    bars[max_attn_idx].set_edgecolor("black")
    bars[max_attn_idx].set_linewidth(1.5)
    bars[max_attn_idx].set_alpha(1.0)

    ax_attn.set_ylabel("Atención", fontsize=8)
    ax_attn.set_xlim(0.5, n + 0.5)
    ax_attn.set_xticks(caps)
    ax_attn.grid(alpha=0.25, axis="y")
    ax_attn.tick_params(labelsize=7)
    ax_attn.annotate(f"cap {max_attn_idx + 1}",
                     xy=(caps[max_attn_idx], d["attention_weights"][max_attn_idx]),
                     xytext=(0, 4), textcoords="offset points",
                     ha="center", fontsize=6, color="black")

    # ── Panel 3: distribución emocional agrupada (stacked bar) ───────────────
    bottoms = np.zeros(n)
    for group, values in real_grouped.items():
        ax_dist.bar(caps, values, bottom=bottoms,
                    color=GROUP_COLORS[group], alpha=0.85,
                    edgecolor="white", linewidth=0.3,
                    label=group)
        bottoms += values

    ax_dist.set_ylabel("Distribución emocional", fontsize=8)
    ax_dist.set_xlabel("Capítulo", fontsize=8)
    ax_dist.set_xlim(0.5, n + 0.5)
    ax_dist.set_xticks(caps)
    ax_dist.grid(alpha=0.25, axis="y")
    ax_dist.tick_params(labelsize=7)


def guardar_subplots_fanfics(
    datos: list,
    output_dir: str = None,
    show_pred: bool = True,
    tropo_filter: int = None,
    max_fanfics: int = None,
):
    """
    Genera y guarda un subplot por fanfic con:
      - Fila 1: arco emocional agrupado (real vs predicho)
      - Fila 2: pesos de atención por capítulo
      - Fila 3: distribución emocional agrupada (stacked bar)

    Parámetros:
        datos:        lista de dicts (salida de extraer_activaciones)
        output_dir:   carpeta de salida (default: configl.FIGURES_DIR/fanfics/)
        show_pred:    superponer predicción del modelo en el arco
        tropo_filter: si int (0/1/2), solo plotea ese tropo
        max_fanfics:  límite de fanfics a plotear (None = todos)

    Archivos generados:
        fanfic_{i:03d}_{titulo}_{tropo}.png
    """
    if output_dir is None:
        output_dir = os.path.join(configl.FIGURES_DIR, "fanfics")
    os.makedirs(output_dir, exist_ok=True)

    # Filtrar
    subset = datos
    if tropo_filter is not None:
        subset = [d for d in datos if d["trope_real"] == tropo_filter]
    if max_fanfics is not None:
        subset = subset[:max_fanfics]

    print(f"Generando {len(subset)} subplots → {output_dir}")

    for i, d in enumerate(subset):
        n      = d["num_chapters"]
        titulo = d["title"][:40].replace(" ", "_").replace("/", "-")
        tropo  = TROPE_NAMES[d["trope_real"]]
        correcto = "✓" if d["trope_real"] == d["trope_pred"] else "✗"

        # Altura dinámica según número de capítulos
        fig_w = max(10, n * 0.9)
        fig   = plt.figure(figsize=(fig_w, 9))
        gs    = gridspec.GridSpec(3, 1, figure=fig,
                                  hspace=0.45,
                                  height_ratios=[2, 1, 1.5])

        ax_arc  = fig.add_subplot(gs[0])
        ax_attn = fig.add_subplot(gs[1], sharex=ax_arc)
        ax_dist = fig.add_subplot(gs[2], sharex=ax_arc)

        plot_fanfic(d, ax_arc, ax_attn, ax_dist, show_pred=show_pred)

        # Título con metadatos
        pred_nombre = TROPE_NAMES[d["trope_pred"]]
        conf        = d["trope_probs"].max()
        fig.suptitle(
            f"{correcto}  «{d['title'][:55]}»\n"
            f"Tropo real: {tropo}  |  "
            f"Predicho: {pred_nombre} ({conf:.0%})  |  "
            f"Fandom: {d['fandom']}  |  {n} capítulos",
            fontsize=9, fontweight="bold", y=0.98
        )

        # Leyenda de grupos (una vez, en el panel del arco)
        patches = [
            mpatches.Patch(color=GROUP_COLORS[g], label=g)
            for g in EMOTION_GROUPS
        ]
        ax_arc.legend(handles=patches, loc="upper right",
                      fontsize=6, ncol=3, framealpha=0.8)

        # Guardar
        fname = f"fanfic_{i:03d}_{titulo}_{tropo}.png"
        fpath = os.path.join(output_dir, fname)
        plt.savefig(fpath, dpi=130, bbox_inches="tight",
                    facecolor="white")
        plt.close(fig)

        if (i + 1) % 10 == 0 or i == len(subset) - 1:
            print(f"  {i+1}/{len(subset)} guardados")

    print(f"\n✓ Subplots guardados en {output_dir}")


def plot_fanfic_interactivo(d: dict, show_pred: bool = True):
    """
    Versión para notebook — muestra el plot inline sin guardar.
    Útil para explorar un fanfic específico.

    Uso:
        fanfic = next(d for d in datos if "título" in d["title"])
        plot_fanfic_interactivo(fanfic)
    """
    n     = d["num_chapters"]
    fig_w = max(10, n * 0.9)
    fig   = plt.figure(figsize=(fig_w, 9))
    gs    = gridspec.GridSpec(3, 1, figure=fig,
                              hspace=0.45,
                              height_ratios=[2, 1, 1.5])

    ax_arc  = fig.add_subplot(gs[0])
    ax_attn = fig.add_subplot(gs[1], sharex=ax_arc)
    ax_dist = fig.add_subplot(gs[2], sharex=ax_arc)

    plot_fanfic(d, ax_arc, ax_attn, ax_dist, show_pred=show_pred)

    tropo    = TROPE_NAMES[d["trope_real"]]
    pred     = TROPE_NAMES[d["trope_pred"]]
    correcto = "✓" if d["trope_real"] == d["trope_pred"] else "✗"
    conf     = d["trope_probs"].max()

    patches = [
        mpatches.Patch(color=GROUP_COLORS[g], label=g)
        for g in EMOTION_GROUPS
    ]
    ax_arc.legend(handles=patches, loc="upper right",
                  fontsize=7, ncol=3, framealpha=0.8)

    fig.suptitle(
        f"{correcto}  «{d['title'][:60]}»\n"
        f"Real: {tropo}  |  Predicho: {pred} ({conf:.0%})  |  "
        f"{d['fandom']}  |  {n} capítulos",
        fontsize=9, fontweight="bold"
    )
    plt.show()


def resumen_visual_tropos(datos: list):
    """
    Vista general: un subplot 3×n donde cada columna es un fanfic
    representativo de cada tropo (el clasificado con mayor confianza).
    Útil para comparar la firma visual entre tropos.
    """
    # Seleccionar el fanfic de mayor confianza por tropo
    representantes = {}
    for t_idx in range(3):
        candidatos = [d for d in datos if d["trope_real"] == t_idx
                      and d["trope_real"] == d["trope_pred"]]
        if not candidatos:
            candidatos = [d for d in datos if d["trope_real"] == t_idx]
        if candidatos:
            representantes[t_idx] = max(candidatos,
                                        key=lambda d: d["trope_probs"][t_idx])

    if not representantes:
        print("No hay datos suficientes para resumen visual.")
        return

    n_tropos = len(representantes)
    fig = plt.figure(figsize=(7 * n_tropos, 9))
    outer_gs = gridspec.GridSpec(1, n_tropos, figure=fig,
                                  wspace=0.35)

    for col, (t_idx, d) in enumerate(representantes.items()):
        inner_gs = gridspec.GridSpecFromSubplotSpec(
            3, 1, subplot_spec=outer_gs[col],
            hspace=0.4, height_ratios=[2, 1, 1.5]
        )
        ax_arc  = fig.add_subplot(inner_gs[0])
        ax_attn = fig.add_subplot(inner_gs[1], sharex=ax_arc)
        ax_dist = fig.add_subplot(inner_gs[2], sharex=ax_arc)

        plot_fanfic(d, ax_arc, ax_attn, ax_dist, show_pred=True)

        conf = d["trope_probs"][t_idx]
        ax_arc.set_title(
            f"{TROPE_NAMES[t_idx]}\n«{d['title'][:35]}»\n"
            f"confianza: {conf:.0%}",
            fontsize=8, fontweight="bold",
            color=TROPE_COLORS[t_idx]
        )

    patches = [
        mpatches.Patch(color=GROUP_COLORS[g], label=g)
        for g in EMOTION_GROUPS
    ]
    fig.legend(handles=patches, loc="lower center", ncol=6,
               fontsize=8, framealpha=0.8,
               bbox_to_anchor=(0.5, 0.01))

    fig.suptitle("Fanfic representativo por tropo\n"
                 "(mayor confianza de clasificación)",
                 fontsize=12, fontweight="bold", y=1.01)

    os.makedirs(configl.FIGURES_DIR, exist_ok=True)
    plt.savefig(f"{configl.FIGURES_DIR}09_representantes_por_tropo.png",
                dpi=130, bbox_inches="tight", facecolor="white")
    plt.show()
    print(f"✓ Guardado: 09_representantes_por_tropo.png")
