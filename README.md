
# Modelo computacional de ToM afectiva en fanfiction

Análisis de la dinámica temporal de la Teoría de la Mente afectiva
en textos de fanfiction mediante RoBERTa-GoEmotions + LSTM bidireccional.

## Integrantes
- Marcela Aguirre Valdez
- Mariana Zúñiga Sánchez

## Estructura del repositorio
- src/                     
config.py            # Configuración del web scraping 
configl.py           # Configuraciones del modelo 
model.py             # Arquitectura de la red neuronal (LSTM / Transformers)
train.py             # Script de entrenamiento y bucles de optimización
feature_extraction.py # Extracción de características y embeddings

scripts/                 
- scraper1.py          # Extractor de datos de fanfics (AO3)
- ao3_auth.py          # Autenticación y manejo de sesiones para el scraper
- supplementer.py      # Script de soporte para completar datos faltantes
- builder.py           # Constructor y procesador del dataset crudo
- nuevo_analysis.py    # Análisis estadístico y generación de heatmaps de emociones
- fanfic_plots.py      # Generador de gráficos de arcos emocionales individuales
- dataset_viz.py       # Visualizaciones descriptivas del corpus general
- pca_lstm_ventanas.py # Análisis de ventanas temporales con reducción dimensional
- pca_trayectorias.py  # Modelado de trayectorias de tropos en el espacio latente
-  poster_figures.py    # Renderizado de figuras en alta calidad para el póster/reporte

notebooks/               # EXPERIMENTACIÓN Y EXPLORACIÓN
- Análisis.ipynb       # Pruebas iniciales de correlaciones y tropos
ao3_pipeline.ipynb   # Pruebas interactivas del flujo de descarga
entrenamiento.ipynb  # Ajuste fino y curvas de aprendizaje del modelo
ROBERTA.ipynb        # Inferencia y pruebas con GoEmotions (SamLowe)

data/                    # DATOS (Solo el índice viaja a Git)
dataset.csv          # Índice general de fanfics, capítulos y metadata
raw/                 # Archivos JSON crudos por capítulo (Ignorado en Git)
features/            # Matrices .pkl con embeddings emocionales (Ignorado en Git)

results/                 # PRODUCTOS FINALIZADOS (Visualizaciones y Métricas)
 *.png                # Gráficos de desempeño, matrices de Pearson y Spearman
grid_search_results.json # Historial de hiperparámetros evaluados
 figures/             # Paneles de figuras exportados listos para análisis
    fanfics/         # Curvas emocionales detalladas por historia
        oneshots/        # Perfiles emocionales de textos de un solo capítulo
