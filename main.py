"""
═══════════════════════════════════════════════════════════════════
Pipeline principal: Modelado computacional de ToM afectiva
en fanfiction mediante RoBERTa + LSTM con atención
 
Ejecutar con: python main.py
 
Pipeline completo:
  1. Cargar features procesados (o procesarlos desde raw)
  2. Entrenar modelo conjunto
  3. Evaluar en test set
  4. Ejecutar análisis de interpretabilidad
  5. Generar todas las figuras
═══════════════════════════════════════════════════════════════════
"""
 
import os
import pickle
import torch
import numpy as np
from torch.utils.data import DataLoader
 
import config
from model import JointModel
from train import FanficDataset, create_dataloaders, JointTrainer
from analysis import ToMInterpreter
 
 
def main():
    # ─── Setup ──────────────────────────────────────────────
    for d in [config.DATA_DIR, config.RAW_DIR, config.FEATURES_DIR,
              config.MODELS_DIR, config.RESULTS_DIR, config.FIGURES_DIR]:
        os.makedirs(d, exist_ok=True)
    
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo: {device}")
    
    # ─── 1. Cargar features ─────────────────────────────────
    features_path = os.path.join(config.FEATURES_DIR, "emotional_features.pkl")
    
    if os.path.exists(features_path):
        print(f"\nCargando features desde {features_path}...")
        with open(features_path, "rb") as f:
            features = pickle.load(f)
    else:
        print("\n⚠ No se encontraron features procesados.")
        print("  Para procesarlos:")
        print("    1. Ejecuta el scraper para obtener los fanfics")
        print("    2. Ejecuta feature_extraction.process_full_dataset(dataset)")
        print("\n  Usando datos sintéticos para demo...")
        features = generate_synthetic_data()
    
    print(f"Total de fanfics: {len(features)}")
    
    # ─── 2. Crear dataloaders ───────────────────────────────
    dataset = FanficDataset(features=features)
    train_loader, val_loader, test_loader = create_dataloaders(dataset)
    
    # ─── 3. Entrenar modelo ─────────────────────────────────
    model = JointModel().to(device)
    print(f"\nModelo: {sum(p.numel() for p in model.parameters()):,} parámetros")
    
    trainer = JointTrainer(model, lambda_reader=1.0, device=device)
    history = trainer.train(train_loader, val_loader)
    
    # ─── 4. Evaluar ─────────────────────────────────────────
    results = trainer.test(test_loader)
    
    # Guardar resultados
    with open(os.path.join(config.RESULTS_DIR, "test_results.pkl"), "wb") as f:
        pickle.dump(results, f)
    
    # ─── 5. Análisis de interpretabilidad ───────────────────
    print("\nEjecutando análisis de interpretabilidad...")
    
    # Usar el dataset completo para análisis (no solo test)
    full_loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    interpreter = ToMInterpreter(model, device=device)
    interpreter.run_full_analysis(full_loader, features)
    
    # ─── 6. Guardar historia de entrenamiento ───────────────
    with open(os.path.join(config.RESULTS_DIR, "training_history.pkl"), "wb") as f:
        pickle.dump(history, f)
    
    print("\n" + "=" * 60)
    print("Pipeline completado exitosamente")
    print(f"  Modelo guardado en: {config.MODELS_DIR}")
    print(f"  Resultados en:      {config.RESULTS_DIR}")
    print(f"  Figuras en:         {config.FIGURES_DIR}")
    print("=" * 60)
 
 
def generate_synthetic_data():
    """Genera datos sintéticos con patrones emocionales realistas por trope."""
    np.random.seed(config.RANDOM_SEED)
    data = []
    
    for fandom in config.FANDOMS:
        for trope_name in config.TROPES:
            for i in range(config.FANFICS_PER_FANDOM_PER_TROPE):
                n_caps = np.random.randint(3, 16)  # 3-15 capítulos
                trope_idx = config.TROPE_TO_IDX[trope_name]
                
                # Base noise
                base = np.random.rand(n_caps, config.FEATURE_DIM) * 0.15
                
                # Patrones específicos por trope
                for c in range(n_caps):
                    p = c / max(n_caps - 1, 1)  # Progresión 0→1
                    
                    if trope_name == "hurt_comfort":
                        base[c, 14] += 0.6 * (1 - p)      # fear
                        base[c, 25] += 0.65 * (1 - p)      # sadness
                        base[c, 16] += 0.4 * (1 - p)       # grief
                        base[c, 17] += 0.6 * p              # joy
                        base[c, 23] += 0.55 * p             # relief
                        base[c, 5]  += 0.5 * p              # caring
                    
                    elif trope_name == "fluff":
                        base[c, 17] += 0.6 + np.random.rand() * 0.1   # joy
                        base[c, 18] += 0.55 + np.random.rand() * 0.1  # love
                        base[c, 1]  += 0.3                              # amusement
                        base[c, 5]  += 0.4                              # caring
                    
                    elif trope_name == "slow_burn":
                        base[c, 8]  += 0.5 * p              # desire
                        base[c, 18] += 0.45 * p             # love
                        base[c, 7]  += 0.35                  # curiosity
                        base[c, 19] += 0.25 * (1 - p)       # nervousness
                        base[c, 13] += 0.3 * p              # excitement
                
                writer_arc = np.clip(base, 0, 1).astype(np.float32)
                
                # Reader arc: correlacionado pero con ruido y lag
                reader_noise = np.random.randn(*writer_arc.shape) * 0.08
                reader_arc = np.clip(writer_arc + reader_noise, 0, 1).astype(np.float32)
                
                data.append({
                    "writer_arc": writer_arc,
                    "reader_arc": reader_arc,
                    "trope": trope_idx,
                    "fandom": fandom,
                    "title": f"{fandom}_{trope_name}_{i:03d}",
                    "num_chapters": n_caps
                })
    
    print(f"Generados {len(data)} fanfics sintéticos")
    return data
 
 
if __name__ == "__main__":
    main()