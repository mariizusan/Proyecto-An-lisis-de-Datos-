"""
═══════════════════════════════════════════════════════════════════
Arquitectura final v3 — Tarea 2 mejorada

Cambios respecto a v2:
  1. Tarea 2 usa AMBOS niveles temporales:
     - WindowLSTM (intra-capítulo): cómo cambia la emoción dentro del cap
     - ArcLSTM (inter-capítulos): contexto narrativo del cap en la obra
     - CommentLSTM: respuesta del lector
     → [256+256+256=768] → fusión → [128] → [28]

  2. Loss ponderada por varianza de cada emoción (en train.py):
     emociones que varían entre tropes pesan más
═══════════════════════════════════════════════════════════════════
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import configl


class ChapterAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1, bias=False)
        )

    def forward(self, hidden_states, mask=None):
        scores = self.attn(hidden_states).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        weights = F.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), hidden_states).squeeze(1)
        return context, weights


class JointModel(nn.Module):
    def __init__(self):
        super().__init__()
        embed_dim = configl.ROBERTA_EMBED_DIM       # 768
        hidden = configl.HIDDEN_DIM                   # 128
        n_layers = configl.NUM_LSTM_LAYERS            # 2
        dropout = configl.DROPOUT                     # 0.3
        bidir = configl.BIDIRECTIONAL                 # True
        n_dir = 2 if bidir else 1
        lstm_out = hidden * n_dir                     # 256

        # ── WindowLSTM: intra-capítulo ──
        self.window_lstm = nn.LSTM(
            input_size=embed_dim, hidden_size=hidden,
            num_layers=1, batch_first=True, bidirectional=bidir
        )

        # ── CommentLSTM: respuesta lectora ──
        self.comment_lstm = nn.LSTM(
            input_size=embed_dim, hidden_size=hidden,
            num_layers=1, batch_first=True, bidirectional=bidir
        )

        # ── ArcLSTM: inter-capítulos ──
        self.arc_lstm = nn.LSTM(
            input_size=lstm_out, hidden_size=hidden,
            num_layers=n_layers, batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=bidir
        )

        # ── Tarea 1: trope (obra completa) ──
        self.chapter_attn = ChapterAttention(lstm_out)
        self.trope_fusion = nn.Sequential(
            nn.Linear(lstm_out + lstm_out, lstm_out),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(lstm_out, hidden),
            nn.ReLU(), nn.Dropout(dropout / 2)
        )
        self.trope_head = nn.Linear(hidden, configl.NUM_TROPES)

        # ── Tarea 2: respuesta emocional (por capítulo) ──
        # Entrada TRIPLE: WindowLSTM + ArcLSTM + CommentLSTM
        self.reader_fusion = nn.Sequential(
            nn.Linear(lstm_out * 3, lstm_out),       # 768 → 256
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(lstm_out, hidden),              # 256 → 128
            nn.ReLU(), nn.Dropout(dropout / 2)
        )
        self.reader_head = nn.Sequential(
            nn.Linear(hidden, configl.NUM_EMOTIONS),  # → 28
        )

    def _lstm_last_hidden(self, lstm, x, lengths=None):
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu().clamp(min=1),
                batch_first=True, enforce_sorted=False
            )
            _, (h_n, _) = lstm(packed)
        else:
            _, (h_n, _) = lstm(x)
        return torch.cat([h_n[-2], h_n[-1]], dim=1)

    def _lstm_with_states(self, lstm, x, lengths=None):
        """
        Como _lstm_last_hidden pero también devuelve todos los hidden states
        intermedios por ventana — para visualizar trayectoria intra-capítulo.

        Returns:
            last:   (B, 256) — último hidden state
            states: (B, n_wins, 256) — hidden state en cada ventana
        """
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu().clamp(min=1),
                batch_first=True, enforce_sorted=False
            )
            output_packed, (h_n, _) = lstm(packed)
            output, _ = nn.utils.rnn.pad_packed_sequence(
                output_packed, batch_first=True
            )
        else:
            output, (h_n, _) = lstm(x)
        last = torch.cat([h_n[-2], h_n[-1]], dim=1)   # (B, 256)
        return last, output                             # (B, 256), (B, n_wins, 256)

    def forward(self, window_embs, window_counts, comment_embs,
                chapter_lengths, return_attention=False, return_hidden=False,
                return_window_states=False):
        B = window_embs.size(0)
        max_caps = window_embs.size(1)
        max_wins = window_embs.size(2)
        device = window_embs.device

        win_vecs, com_vecs = [], []
        all_window_states  = []

        for c in range(max_caps):
            wins   = window_embs[:, c, :, :]
            n_wins = window_counts[:, c].clamp(min=1)

            if return_window_states:
                last, states = self._lstm_with_states(self.window_lstm, wins, n_wins)
                win_vecs.append(last)
                all_window_states.append(states)
            else:
                win_vecs.append(self._lstm_last_hidden(self.window_lstm, wins, n_wins))

            c_emb = comment_embs[:, c, :, :]
            com_vecs.append(self._lstm_last_hidden(self.comment_lstm, c_emb))

        win_seq = torch.stack(win_vecs, dim=1)  # (B, max_caps, 256)
        com_seq = torch.stack(com_vecs, dim=1)  # (B, max_caps, 256)

        # ── ArcLSTM sobre capítulos ──
        packed = nn.utils.rnn.pack_padded_sequence(
            win_seq, chapter_lengths.cpu().clamp(min=1),
            batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.arc_lstm(packed)
        arc_hidden, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=max_caps
        )  # (B, max_caps, 256)

        cap_mask = torch.arange(max_caps, device=device).unsqueeze(0)
        cap_mask = (cap_mask < chapter_lengths.unsqueeze(1)).float()

        # ══ TAREA 1: trope ══
        arc_context, attn_weights = self.chapter_attn(arc_hidden, cap_mask)
        com_avg = (com_seq * cap_mask.unsqueeze(2)).sum(1) / cap_mask.sum(1, keepdim=True)
        trope_fused = self.trope_fusion(
            torch.cat([arc_context, com_avg], dim=1)
        )
        trope_logits = self.trope_head(trope_fused)

        # ══ TAREA 2: respuesta emocional — TRIPLE temporal ══
        reader_input = torch.cat([
            win_seq,      # intra-capítulo (cómo cambia dentro del cap)
            arc_hidden,   # inter-capítulos (dónde está este cap en la obra)
            com_seq       # respuesta del lector
        ], dim=2)         # (B, max_caps, 768)

        reader_fused = self.reader_fusion(reader_input)
        reader_pred = self.reader_head(reader_fused)

        output = {"trope_logits": trope_logits, "reader_pred": reader_pred}
        if return_attention:
            output["attention_weights"] = attn_weights
        if return_hidden:
            output["hidden_states"] = arc_hidden
            output["window_vecs"] = win_seq
        if return_window_states and all_window_states:
            # Lista de tensores (B, n_wins, 256), uno por capítulo
            output["window_states"] = all_window_states
        return output


if __name__ == "__main__":
    B, C, W = 4, 8, 20
    we = torch.randn(B, C, W, 768)
    wc = torch.randint(5, W+1, (B, C))
    ce = torch.randn(B, C, 1, 768)
    cl = torch.tensor([8, 5, 1, 8])

    model = JointModel()
    out = model(we, wc, ce, cl, return_attention=True, return_hidden=True)
    print(f"Trope logits:    {out['trope_logits'].shape}")
    print(f"Reader pred:     {out['reader_pred'].shape}")
    print(f"Attention:       {out['attention_weights'].shape}")
    print(f"Parámetros:      {sum(p.numel() for p in model.parameters()):,}")
