"""
═══════════════════════════════════════════════════════════════════
Entrenamiento v3 — Loss ponderada por varianza emocional
 
Cambio clave: MSE de la Tarea 2 se pondera por la varianza de cada
emoción. Emociones que varían entre tropes (sadness, amusement, neutral)
pesan más. Emociones constantes (grief, pride, relief ≈ 0) pesan menos.
═══════════════════════════════════════════════════════════════════
"""
 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pickle, os
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from collections import defaultdict
import configl
from model import JointModel

 
class FanficDataset(Dataset):
    def __init__(self, features):
        self.data = features
 
    def __len__(self):
        return len(self.data)
 
    def __getitem__(self, idx):
        item = self.data[idx]
        n_caps = min(item["num_chapters"], configl.MAX_SEQ_LEN)
        max_caps = configl.MAX_SEQ_LEN
        max_wins = configl.MAX_WINDOWS_PER_CHAPTER
        edim = configl.ROBERTA_EMBED_DIM
 
        win_pad = np.zeros((max_caps, max_wins, edim), dtype=np.float32)
        win_counts = np.zeros(max_caps, dtype=np.int64)
        for c in range(n_caps):
            emb = item["chapter_embeddings"][c]
            nw = min(emb.shape[0], max_wins)
            win_pad[c, :nw, :] = emb[:nw]
            win_counts[c] = nw
 
        com_pad = np.zeros((max_caps, 1, edim), dtype=np.float32)
        if "comment_embeddings" in item:
            for c in range(n_caps):
                if c < len(item["comment_embeddings"]):
                    emb_com = item["comment_embeddings"][c]
                    # Validación para evitar crash si el capítulo no tiene comentarios
                    if len(emb_com) > 0:
                        com_pad[c] = emb_com[:1]
 
        reader_pad = np.zeros((max_caps, configl.NUM_EMOTIONS), dtype=np.float32)
        ra = item["reader_arc"][:n_caps]
        reader_pad[:len(ra)] = ra
 
        return {
            "window_embs": torch.FloatTensor(win_pad),
            "window_counts": torch.LongTensor(win_counts),
            "comment_embs": torch.FloatTensor(com_pad),
            "reader_arc": torch.FloatTensor(reader_pad),
            "trope": torch.LongTensor([item["trope"]])[0],
            "length": torch.LongTensor([n_caps])[0]
        }
 
 
def compute_emotion_weights(features):
    """
    Calcula pesos por emoción basados en su varianza.
    Emociones con más varianza → más peso en la loss.
    Emociones con varianza ~0 → peso mínimo (no desperdician capacidad).
    """
    all_arcs = np.concatenate([f["reader_arc"] for f in features], axis=0)
    variances = all_arcs.var(axis=0)  # (28,)
 
    # Normalizar: la emoción con mayor varianza pesa 1.0
    weights = variances / (variances.max() + 1e-8)
 
    # Piso mínimo para no ignorar completamente ninguna
    weights = np.clip(weights, 0.01, 1.0)
 
    print(f"Pesos emocionales (top 5 señal):")
    labels = configl.EMOTION_LABELS
    ranked = weights.argsort()[::-1]
    for i in range(8):
        idx = ranked[i]
        print(f"  {labels[idx]:15s}: {weights[idx]:.3f}")
    print(f"  ... ({(weights < 0.05).sum()} emociones con peso <0.05)")
 
    return torch.FloatTensor(weights)
 
 
def create_dataloaders(dataset, batch_size=configl.BATCH_SIZE, seed=configl.RANDOM_SEED):
    torch.manual_seed(seed)
    n = len(dataset)
    n_train = int(n * configl.TRAIN_SPLIT)
    n_val = int(n * configl.VAL_SPLIT)
    n_test = n - n_train - n_val
    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test])
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True),
            DataLoader(val_ds, batch_size=batch_size),
            DataLoader(test_ds, batch_size=batch_size))
 
 
class Trainer:
    def __init__(self, model, emotion_weights, alpha=0.5,
                 lr=configl.LEARNING_RATE, wd=configl.WEIGHT_DECAY, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.alpha = alpha
 
        # Pesos por emoción para Tarea 2
        self.emotion_weights = emotion_weights.to(self.device)  # (28,)
 
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5)
        self.trope_loss_fn = nn.CrossEntropyLoss()
        self.history = defaultdict(list)
 
    def _to_device(self, batch):
        return {k: v.to(self.device) for k, v in batch.items()}
 
    def _reader_loss_weighted(self, pred, target, lengths):
        """
        KL divergence entre distribución predicha y real (Corregida Indentación).
        pred:   logits crudos (B, max_caps, 28)
        target: distribución real (B, max_caps, 28) — suma ~1
        """
        max_len = pred.size(1)
        cap_mask = torch.arange(max_len, device=pred.device).unsqueeze(0)
        cap_mask = (cap_mask < lengths.unsqueeze(1)).unsqueeze(2).float()
    
        # Suavizar target para evitar log(0)
        epsilon = 1e-8
        target_smooth = target + epsilon
        target_smooth = target_smooth / target_smooth.sum(dim=-1, keepdim=True)
    
        # Log-softmax de la predicción (distribución predicha)
        log_pred = torch.log_softmax(pred, dim=-1)
    
        # KL divergence: Σ target * (log(target) - log(pred))
        kl = torch.nn.functional.kl_div(
            log_pred, target_smooth,
            reduction="none"
        )  # (B, max_caps, 28)
    
        # Ponderar por varianza emocional
        kl_weighted = kl * self.emotion_weights.unsqueeze(0).unsqueeze(0)
    
        # Promediar solo capítulos reales
        loss = (kl_weighted * cap_mask).sum() / (cap_mask.sum() * pred.size(2))
        return loss
 
    def _step(self, batch):
        b = self._to_device(batch)
        out = self.model(b["window_embs"], b["window_counts"],
                         b["comment_embs"], b["length"])
        l_trope = self.trope_loss_fn(out["trope_logits"], b["trope"])
        l_reader = self._reader_loss_weighted(out["reader_pred"], b["reader_arc"], b["length"])
        loss = self.alpha * l_trope + (1 - self.alpha) * l_reader
        preds = out["trope_logits"].argmax(1).cpu().numpy()
        labels = b["trope"].cpu().numpy()
        return loss, l_trope.item(), l_reader.item(), preds, labels
 
    def train_epoch(self, dl):
        self.model.train()
        t = {"loss": 0, "trope": 0, "reader": 0}
        all_p, all_l = [], []
        for batch in dl:
            self.optimizer.zero_grad()
            loss, lt, lr, p, l = self._step(batch)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            t["loss"] += loss.item(); t["trope"] += lt; t["reader"] += lr
            all_p.extend(p); all_l.extend(l)
        n = len(dl)
        return {"loss": t["loss"]/n, "trope_loss": t["trope"]/n,
                "reader_loss": t["reader"]/n,
                "accuracy": accuracy_score(all_l, all_p),
                "f1": f1_score(all_l, all_p, average="macro")}
 
    @torch.no_grad()
    def evaluate(self, dl):
        self.model.eval()
        t = {"loss": 0, "trope": 0, "reader": 0}
        all_p, all_l = [], []
        for batch in dl:
            loss, lt, lr, p, l = self._step(batch)
            t["loss"] += loss.item(); t["trope"] += lt; t["reader"] += lr
            all_p.extend(p); all_l.extend(l)
        n = len(dl)
        return {"loss": t["loss"]/n, "trope_loss": t["trope"]/n,
                "reader_loss": t["reader"]/n,
                "accuracy": accuracy_score(all_l, all_p),
                "f1": f1_score(all_l, all_p, average="macro")}
 
    def fit(self, train_dl, val_dl, epochs=configl.NUM_EPOCHS,
            patience=configl.PATIENCE, save_dir=configl.MODELS_DIR):
        os.makedirs(save_dir, exist_ok=True)
        best_val = float("inf"); wait = 0
        for ep in range(1, epochs + 1):
            tm = self.train_epoch(train_dl)
            vm = self.evaluate(val_dl)
            self.scheduler.step(vm["loss"])
            for k, v in tm.items(): self.history[f"train_{k}"].append(v)
            for k, v in vm.items(): self.history[f"val_{k}"].append(v)
            lr = self.optimizer.param_groups[0]["lr"]
            print(f"Ep {ep:3d}/{epochs} | "
                  f"Train {tm['loss']:.4f} (T:{tm['trope_loss']:.4f} R:{tm['reader_loss']:.4f}) "
                  f"Acc:{tm['accuracy']:.3f} F1:{tm['f1']:.3f} | "
                  f"Val {vm['loss']:.4f} Acc:{vm['accuracy']:.3f} F1:{vm['f1']:.3f} | "
                  f"LR:{lr:.1e}")
            if vm["loss"] < best_val:
                best_val = vm["loss"]; wait = 0
                torch.save(self.model.state_dict(), os.path.join(save_dir, "best.pt"))
                print(f"  ✓ Mejor modelo (val={best_val:.4f})")
            else:
                wait += 1
                if wait >= patience:
                    print(f"\n  Early stopping ep {ep}")
                    break
        self.model.load_state_dict(
            torch.load(os.path.join(save_dir, "best.pt"), weights_only=True))
        return dict(self.history)
 
    @torch.no_grad()
    def test(self, dl):
        self.model.eval()
        all_p, all_l, all_mse = [], [], []
        for batch in dl:
            b = self._to_device(batch)
            out = self.model(b["window_embs"], b["window_counts"],
                             b["comment_embs"], b["length"])
            all_p.extend(out["trope_logits"].argmax(1).cpu().numpy())
            all_l.extend(b["trope"].cpu().numpy())
            
            # MSE por muestra (sin ponderación, para comparar con baseline lineal de regresión)
            max_len = out["reader_pred"].size(1)
            mask = torch.arange(max_len, device=self.device).unsqueeze(0)
            mask = (mask < b["length"].unsqueeze(1)).float()
            mse = ((out["reader_pred"] - b["reader_arc"])**2).mean(dim=2)
            mse = (mse * mask).sum(dim=1) / mask.sum(dim=1)
            all_mse.extend(mse.cpu().numpy())
 
        names = [configl.IDX_TO_TROPE[i] for i in range(configl.NUM_TROPES)]
        cm = confusion_matrix(all_l, all_p)
        print("\n" + "="*60)
        print("RESULTADOS TEST")
        print("="*60)
        print("\nTarea 1 — Clasificación de trope:")
        print(classification_report(all_l, all_p, target_names=names))
        print(f"Matriz de confusión:\n{cm}")
        print(f"\nTarea 2 — Predicción emocional:")
        print(f"  MSE promedio: {np.mean(all_mse):.4f}")
        print(f"  MSE std:      {np.std(all_mse):.4f}")
        return {"report": classification_report(all_l, all_p, target_names=names, output_dict=True),
                "confusion_matrix": cm,
                "reader_mse": np.mean(all_mse)}
 
 
def train(pkl_path="data/features/emotional_features.pkl",
          alpha=0.5, device_str=None):
    with open(pkl_path, "rb") as f:
        features = pickle.load(f)
    print(f"Fanfics: {len(features)}")
 
    # Calcular pesos emocionales
    emotion_weights = compute_emotion_weights(features)
 
    ds = FanficDataset(features)
    train_dl, val_dl, test_dl = create_dataloaders(ds)
 
    model = JointModel()
    print(f"Parámetros: {sum(p.numel() for p in model.parameters()):,}")
 
    device = device_str or ("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(model, emotion_weights, alpha=alpha, device=device)
 
    print(f"\nEntrenando (α={alpha}, device={device})...\n")
    history = trainer.fit(train_dl, val_dl)
 
    print("\nEvaluando en test...")
    metrics = trainer.test(test_dl)
 
    os.makedirs(configl.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(configl.RESULTS_DIR, "history.pkl"), "wb") as f:
        pickle.dump(history, f)
 
    return model, history, metrics
 
 
if __name__ == "__main__":
    train()