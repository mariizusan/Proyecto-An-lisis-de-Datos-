3
"""
═══════════════════════════════════════════════════════════════════
Configuración del proyecto: Modelado computacional de ToM afectiva
en fanfiction mediante RoBERTa + LSTM con atención
═══════════════════════════════════════════════════════════════════
"""
 
# ─── Dataset ────────────────────────────────────────────────────
FANDOMS = ["One Direction", "BTS", "Harry Potter"]
TROPES = ["hurt_comfort", "fluff", "slow_burn"]
NUM_TROPES = len(TROPES)  # 3
TROPE_TO_IDX = {t: i for i, t in enumerate(TROPES)}
IDX_TO_TROPE = {i: t for t, i in TROPE_TO_IDX.items()}
 
# Criterios de filtrado
MIN_KUDOS = 500
MIN_CHAPTERS = 1
MAX_CHAPTERS = 15
MIN_WORD_COUNT = 2000
MAX_WORD_COUNT = 6000
MIN_COMMENTS_PER_CHAPTER = 15
MIN_COMMENT_WORDS = 15
FANFICS_PER_FANDOM_PER_TROPE = 30
 
# ─── RoBERTa / GoEmotions ──────────────────────────────────────
ROBERTA_MODEL = "SamLowe/roberta-base-go_emotions"
ROBERTA_MAX_TOKENS = 512
ROBERTA_EMBED_DIM = 768            # Dimensión del embedding [CLS]
CHUNK_OVERLAP_RATIO = 0.10         # 10% de solapamiento entre ventanas
CHUNK_OVERLAP_TOKENS = int(ROBERTA_MAX_TOKENS * CHUNK_OVERLAP_RATIO)  # ~51
 
EMOTION_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness",
    "optimism", "pride", "realization", "relief", "remorse",
    "sadness", "surprise", "neutral"
]
NUM_EMOTIONS = len(EMOTION_LABELS)  # 28
 
# ─── Representaciones ──────────────────────────────────────────
# Capítulos: cada capítulo → [n_ventanas, 768] (embeddings crudos)
# Se agregan a nivel de capítulo con atención antes de la LSTM
CHAPTER_FEATURE_DIM = ROBERTA_EMBED_DIM  # 768
 
# Comentarios: cada comentario → etiqueta de emoción dominante
# Por capítulo se agrega como distribución normalizada → [28]
COMMENT_FEATURE_DIM = NUM_EMOTIONS  # 28
 
# ─── Modelo ────────────────────────────────────────────────────
# Encoder de ventanas (agrupa ventanas → 1 vector por capítulo)
WINDOW_ATTN_DIM = 256               # Proyección interna de atención de ventanas
 
# LSTM de secuencia (procesa la secuencia de capítulos)
HIDDEN_DIM = 128
NUM_LSTM_LAYERS = 2
DROPOUT = 0.3
BIDIRECTIONAL = True
USE_ATTENTION = True
MAX_SEQ_LEN = 25                    # Máximo de capítulos (padding)
MAX_WINDOWS_PER_CHAPTER = 60       # Padding de ventanas por capítulo
 
# ─── Entrenamiento ──────────────────────────────────────────────
BATCH_SIZE = 32                     # Más bajo por el tamaño de los embeddings
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 50
PATIENCE = 10
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42
 
# ─── Rutas ──────────────────────────────────────────────────────
DATA_DIR = "data/"
RAW_DIR = DATA_DIR + "raw/"
FEATURES_DIR = DATA_DIR + "features/"
MODELS_DIR = "models/"
RESULTS_DIR = "results/"
FIGURES_DIR = "results/figures/"
