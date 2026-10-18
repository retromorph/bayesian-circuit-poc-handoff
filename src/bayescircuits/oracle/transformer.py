"""Frozen-transformer head-ablation oracle (TransformerLens ``hook_z``).

For a batch of masks over a fixed candidate head list and a fixed prompt batch, every
(mask, prompt) pair is run through the frozen model with the selected candidate heads'
``blocks.{l}.attn.hook_z`` output set to zero at **all token positions**. Non-candidate heads,
MLPs, embeddings and residual paths are untouched. Returns per-prompt
``d(x, m) = s(F(x)) - s(F^do(m)(x))`` measured by actually running the model.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import torch

ScoreFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]  # (logits[B,T,V], pos[B], prompt_id[B]) -> score[B]


class HeadAblationRunner:
    def __init__(
        self,
        model,
        candidate_heads: Sequence[tuple[int, int]],
        tokens: torch.Tensor,
        score_fn: ScoreFn,
        score_pos: torch.Tensor,
        microbatch: int = 512,
    ):
        self.model = model
        self.heads = [(int(layer), int(h)) for layer, h in candidate_heads]
        self.tokens = tokens
        self.score_fn = score_fn
        self.score_pos = score_pos
        self.microbatch = microbatch
        self.device = tokens.device
        self.n_layers = model.cfg.n_layers
        self.n_heads = model.cfg.n_heads
        self.layer_of = torch.tensor([h[0] for h in self.heads], device=self.device)
        self.head_of = torch.tensor([h[1] for h in self.heads], device=self.device)
        self.n_forward = 0
        self.all_ids = torch.arange(len(tokens), device=self.device)
        with torch.inference_mode():
            self.clean = self._run_keep(
                torch.ones(len(tokens), self.n_layers, self.n_heads, device=self.device), tokens, score_pos, self.all_ids
            )

    def keep_tensor(self, masks: np.ndarray) -> torch.Tensor:
        """(b, p) ablation masks -> (b, L, H) multiplicative keep factors (1 = keep)."""
        m = torch.as_tensor(np.asarray(masks), dtype=torch.float32, device=self.device)
        keep = torch.ones(m.shape[0], self.n_layers, self.n_heads, device=self.device)
        keep[:, self.layer_of, self.head_of] = 1.0 - m
        return keep

    def _run_keep(self, keep: torch.Tensor, toks: torch.Tensor, pos: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        hooks = []
        for layer in range(self.n_layers):
            kl = keep[:, layer, :]
            if torch.all(kl == 1):
                continue

            def hook(z, hook, kl=kl):
                return z * kl[:, None, :, None]

            hooks.append((f"blocks.{layer}.attn.hook_z", hook))
        logits = self.model.run_with_hooks(toks, fwd_hooks=hooks) if hooks else self.model(toks)
        self.n_forward += toks.shape[0]
        return self.score_fn(logits, pos, ids)

    @torch.inference_mode()
    def scores(self, masks: np.ndarray, prompt_idx: np.ndarray | None = None) -> np.ndarray:
        """Masked scores, shape (b, n_prompts)."""
        ids = self.all_ids if prompt_idx is None else torch.as_tensor(prompt_idx, device=self.device)
        toks, pos = self.tokens[ids], self.score_pos[ids]
        n = toks.shape[0]
        keep = self.keep_tensor(masks)  # (b, L, H)
        b = keep.shape[0]
        flat_keep = keep.repeat_interleave(n, dim=0)
        flat_tok = toks.repeat(b, 1)
        flat_pos = pos.repeat(b)
        flat_ids = ids.repeat(b)
        mb = self.microbatch
        res = []
        for s in range(0, b * n, mb):
            res.append(self._run_keep(flat_keep[s : s + mb], flat_tok[s : s + mb], flat_pos[s : s + mb], flat_ids[s : s + mb]))
        out = torch.cat(res).reshape(b, n)
        return out.float().cpu().numpy()

    def effects(self, masks: np.ndarray, prompt_idx: np.ndarray | None = None) -> np.ndarray:
        """d(x, m) = s(F(x)) - s(F^do(m)(x)), shape (b, n_prompts)."""
        clean = self.clean if prompt_idx is None else self.clean[torch.as_tensor(prompt_idx, device=self.device)]
        return clean.float().cpu().numpy()[None, :] - self.scores(masks, prompt_idx)


def last_token_margin(correct: torch.Tensor, wrong: torch.Tensor) -> ScoreFn:
    """Score = logit[correct[id]] - logit[wrong[id]] at the per-prompt score position."""

    def fn(logits: torch.Tensor, pos: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        rows = torch.arange(logits.shape[0], device=logits.device)
        lg = logits[rows, pos]  # (B, V)
        return lg[rows, correct[ids]] - lg[rows, wrong[ids]]

    return fn
