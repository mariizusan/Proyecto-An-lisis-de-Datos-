"""
═══════════════════════════════════════════════════════════════════
Extracción de representaciones con RoBERTa-GoEmotions
 
Dos modos de operación:
  1. CAPÍTULOS → embeddings contextuales [CLS] de 768 dimensiones
     por ventana deslizante. Cada capítulo queda como [n_ventanas, 768].
     Los pesos de RoBERTa se mantienen CONGELADOS.
 
  2. COMENTARIOS → etiqueta de emoción dominante por comentario
     usando la cabeza de clasificación de GoEmotions (28 categorías).
     Se agrega por capítulo como distribución normalizada [28].
═══════════════════════════════════════════════════════════════════
"""
 
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict, Tuple
import config
 
 
class EmotionExtractor:
    def __init__(self, model_name: str = config.ROBERTA_MODEL, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Cargando {model_name} en {self.device}...")
 
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, output_hidden_states=True
        )
        self.model.to(self.device)
        self.model.eval()
 
        # Congelar todos los pesos
        for param in self.model.parameters():
            param.requires_grad = False
 
        self.emotion_labels = config.EMOTION_LABELS
 
    # ═══════════════════════════════════════════════════════════
    # CAPÍTULOS: embeddings [CLS] por ventana deslizante
    # ═══════════════════════════════════════════════════════════
 
    def _chunk_text(self, text: str) -> List[str]:
        """
        Ventanas deslizantes de 512 tokens con 10% de solapamiento.
        Preserva continuidad semántica en los bordes.
        """
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        max_chunk = config.ROBERTA_MAX_TOKENS - 2  # Reservar [CLS] y [SEP]
        overlap = config.CHUNK_OVERLAP_TOKENS
        step = max_chunk - overlap
 
        chunks = []
        for start in range(0, len(tokens), step):
            chunk_tokens = tokens[start:start + max_chunk]
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            chunks.append(chunk_text)
            if start + max_chunk >= len(tokens):
                break
 
        return chunks if chunks else [text]
 
    @torch.no_grad()
    def _extract_cls_embedding(self, text: str) -> np.ndarray:
        """
        Extrae el embedding [CLS] de la última capa oculta de RoBERTa.
        Output: vector de 768 dimensiones.
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=config.ROBERTA_MAX_TOKENS,
            padding=True
        ).to(self.device)
 
        outputs = self.model(**inputs)
        # hidden_states[-1] = última capa, [0] = batch, [0] = token [CLS]
        cls_embedding = outputs.hidden_states[-1][0, 0, :].cpu().numpy()
        return cls_embedding  # [768]
 
    def extract_chapter_embeddings(self, chapter_text: str) -> np.ndarray:
        """
        Capítulo completo → secuencia de embeddings por ventana.
 
        Returns:
            np.ndarray de shape (n_ventanas, 768)
        """
        chunks = self._chunk_text(chapter_text)
        embeddings = [self._extract_cls_embedding(chunk) for chunk in chunks]
        return np.array(embeddings)  # [n_ventanas, 768]
 
    def extract_chapter_arc(self, chapters: List[str]) -> List[np.ndarray]:
        """
        Todos los capítulos de un fanfic → lista de matrices de embeddings.
 
        Returns:
            Lista de np.ndarray, cada uno de shape (n_ventanas_i, 768)
            (n_ventanas varía por capítulo según su longitud)
        """
        arc = []
        for i, chapter in enumerate(chapters):
            emb = self.extract_chapter_embeddings(chapter)
            arc.append(emb)
            print(f"    Cap {i+1}/{len(chapters)}: {len(chapter.split())} palabras → {emb.shape[0]} ventanas")
        return arc
 
    # ═══════════════════════════════════════════════════════════
    # COMENTARIOS: emoción dominante por comentario
    # ═══════════════════════════════════════════════════════════
 
    @torch.no_grad()
    def _classify_comment(self, text: str) -> int:
        """
        Clasifica un comentario → índice de la emoción dominante.
        Usa sigmoid (GoEmotions) y toma el argmax.
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=config.ROBERTA_MAX_TOKENS,
            padding=True
        ).to(self.device)
 
        outputs = self.model(**inputs)
        probs = torch.sigmoid(outputs.logits).cpu().numpy()[0]
        return int(probs.argmax())
 
    def extract_comment_distribution(self, comments: List[str]) -> np.ndarray:
        """
        Lista de comentarios de un capítulo → distribución emocional.
 
        Cada comentario se clasifica con su emoción dominante.
        Se cuenta la frecuencia de cada emoción y se normaliza.
 
        Returns:
            np.ndarray de shape (28,) — distribución normalizada
        """
        if not comments:
            return np.zeros(config.NUM_EMOTIONS, dtype=np.float32)
 
        counts = np.zeros(config.NUM_EMOTIONS, dtype=np.float32)
        for comment in comments:
            emo_idx = self._classify_comment(comment)
            counts[emo_idx] += 1
 
        # Normalizar a distribución de probabilidad
        total = counts.sum()
        if total > 0:
            counts /= total
 
        return counts  # [28]
 
    def extract_comment_arc(
        self, comments_by_chapter: List[List[str]]
    ) -> np.ndarray:
        """
        Comentarios por capítulo → arco emocional del lector.
 
        Returns:
            np.ndarray de shape (n_chapters, 28)
        """
        arc = []
        for i, chapter_comments in enumerate(comments_by_chapter):
            dist = self.extract_comment_distribution(chapter_comments)
            top_emo = self.emotion_labels[dist.argmax()] if dist.sum() > 0 else "none"
            arc.append(dist)
            print(f"    Comentarios cap {i+1}: {len(chapter_comments)} → emoción dominante: {top_emo}")
        return np.array(arc)  # [n_chapters, 28]
 
 
def process_full_dataset(dataset: List[Dict], output_dir: str = config.FEATURES_DIR):
    """
    Procesa el dataset completo.
 
    Cada fanfic produce:
      - chapter_embeddings: lista de arrays [n_ventanas_i, 768] por capítulo
      - reader_arc: array [n_chapters, 28] — distribución emocional de comentarios
      - trope, fandom, title, num_chapters
    """
    import os
    import pickle
    os.makedirs(output_dir, exist_ok=True)
 
    extractor = EmotionExtractor()
    all_features = []
 
    for idx, fanfic in enumerate(dataset):
        print(f"\n[{idx+1}/{len(dataset)}] {fanfic['title'][:60]}")
        print(f"  {fanfic['fandom']}/{fanfic['trope']} | {len(fanfic['chapters'])} caps")
 
        # Embeddings de capítulos: lista de [n_ventanas, 768]
        print("  → Embeddings de capítulos...")
        chapter_embeddings = extractor.extract_chapter_arc(fanfic["chapters"])
 
        # Distribución emocional de comentarios: [n_chapters, 28]
        print("  → Clasificando comentarios...")
        reader_arc = extractor.extract_comment_arc(fanfic["comments_by_chapter"])
 
        features = {
            "chapter_embeddings": chapter_embeddings,  # lista de arrays
            "reader_arc": reader_arc,                   # [n_caps, 28]
            "trope": config.TROPE_TO_IDX[fanfic["trope"]],
            "fandom": fanfic["fandom"],
            "title": fanfic["title"],
            "num_chapters": len(fanfic["chapters"])
        }
        all_features.append(features)
 
        # Checkpoint cada 25 fanfics
        if (idx + 1) % 25 == 0:
            ckpt_path = os.path.join(output_dir, "checkpoint_features.pkl")
            with open(ckpt_path, "wb") as f:
                pickle.dump(all_features, f)
            print(f"  💾 Checkpoint guardado ({idx+1}/{len(dataset)})")
 
    # Guardar features finales
    final_path = os.path.join(output_dir, "emotional_features.pkl")
    with open(final_path, "wb") as f:
        pickle.dump(all_features, f)
    print(f"\n✓ Features guardados en {final_path}")
 
    return all_features