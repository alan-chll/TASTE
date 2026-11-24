from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(in_dim: int, hidden: int, out_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, out_dim),
    )


def _n_heads(dim: int, want: int = 4) -> int:
    for h in (want, 8, 6, 4, 2, 1):
        if h <= dim and dim % h == 0:
            return h
    return 1


class TinyEncoder(nn.Module):
    """Hash bag-of-words encoder for logic tests. No pretrained weights."""

    def __init__(self, dim: int = 32, vocab: int = 4096):
        super().__init__()
        self.dim = dim
        self.vocab = vocab
        self.emb = nn.EmbeddingBag(vocab, dim, mode="mean")

    def _ids(self, text: str, device) -> torch.Tensor:
        import hashlib

        toks = (text or "").lower().split() or ["<empty>"]
        ids = []
        for t in toks:
            h = int(hashlib.md5(t.encode("utf-8")).hexdigest(), 16) % self.vocab
            ids.append(h)
        return torch.tensor(ids, dtype=torch.long, device=device)

    def forward(self, texts: List[str]) -> torch.Tensor:
        device = self.emb.weight.device
        vecs = []
        for t in texts:
            idx = self._ids(t, device)
            off = torch.zeros(1, dtype=torch.long, device=device)
            vecs.append(self.emb(idx, off).squeeze(0))
        h = torch.stack(vecs, dim=0)
        return F.normalize(h, p=2, dim=-1)


class HFEncoder(nn.Module):
    def __init__(self, name: str, max_length: int, freeze: bool = False):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name)
        self.dim = int(self.model.config.hidden_size)
        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False

    def forward(self, texts: List[str]) -> torch.Tensor:
        device = next(self.model.parameters()).device
        tok = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        tok = {k: v.to(device) for k, v in tok.items()}
        hidden = self.model(**tok).last_hidden_state
        mask = tok["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        return F.normalize(pooled, p=2, dim=-1)


class TASTE(nn.Module):
    """Enc_theta + candidate self-attention + MLP_sel + MLP_stop. Action 0 = STOP."""

    def __init__(
        self,
        encoder: nn.Module,
        enc_dim: int,
        m_max: int,
        hidden: int = 256,
        dropout: float = 0.0,
        attn_heads: int = 4,
        mask_tau_neg: bool = False,
        skip_attn: bool = False,
        zero_time_feats: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.enc_dim = int(enc_dim)
        self.m_max = int(m_max)
        # Paper: no hard filter on temporal relevance. Keep flag for legacy
        # checkpoints but default/release config leaves it False.
        self.mask_tau_neg = bool(mask_tau_neg)
        self.skip_attn = bool(skip_attn)
        self.zero_time_feats = bool(zero_time_feats)
        extra = 1 + m_max + 1
        n_heads = _n_heads(enc_dim, int(attn_heads))
        self.cand_fuse = nn.Linear(enc_dim + extra, enc_dim)
        self.cand_attn = nn.MultiheadAttention(
            enc_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.cand_ln = nn.LayerNorm(enc_dim)
        # original sel feats + attended context + dup + complementary mass
        sel_in = 1 + m_max + 1 + enc_dim + m_max + enc_dim + 2
        # original stop feats + remaining complementary mean/max
        stop_in = enc_dim + m_max + 1 + 2 + 2
        self.mlp_sel = _mlp(sel_in, hidden, 1, dropout)
        self.mlp_stop = _mlp(stop_in, hidden, 1, dropout)

    def encode_batch(self, mc: List[str], texts: List[List[str]]) -> Tuple[torch.Tensor, torch.Tensor]:
        b = len(mc)
        n = len(texts[0])
        flat = list(mc)
        for row in texts:
            flat.extend(row)
        h = self.encoder(flat)
        h_mc = h[:b]
        h_e = h[b:].view(b, n, -1)
        s_sem = (h_e * h_mc.unsqueeze(1)).sum(dim=-1)
        return h_e, s_sem

    def cand_context(
        self,
        h_e: torch.Tensor,
        s_sem: torch.Tensor,
        tau: torch.Tensor,
        m_time: torch.Tensor,
        cand_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Pool-level context. Independent of S_t; compute once per question, reuse every step."""
        if self.zero_time_feats:
            tau = torch.zeros_like(tau)
            m_time = torch.zeros_like(m_time)
        extra = torch.cat([s_sem.unsqueeze(-1), tau, m_time.unsqueeze(-1)], dim=-1)
        x = self.cand_fuse(torch.cat([h_e, extra], dim=-1))
        if self.skip_attn:
            h_ctx = self.cand_ln(x)
        else:
            pad = cand_mask <= 0
            all_pad = pad.all(dim=-1)
            if bool(all_pad.any()):
                pad = pad.clone()
                pad[all_pad, 0] = False
            attn_out, _ = self.cand_attn(x, x, x, key_padding_mask=pad)
            h_ctx = self.cand_ln(x + attn_out)
        h_ctx = h_ctx * (cand_mask > 0).unsqueeze(-1).to(h_ctx.dtype)
        return h_ctx

    def tau_max_over_m(self, tau: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        """Per-candidate max_k tau^k over the real M TC segments, not m_max pads."""
        idx = torch.arange(self.m_max, device=tau.device).view(1, 1, -1)
        real = idx < M.to(device=tau.device).view(-1, 1, 1)
        filled = tau.masked_fill(~real, float("-inf"))
        tmax = filled.max(dim=-1).values
        return torch.where(torch.isfinite(tmax), tmax, torch.zeros_like(tmax))

    def legal_remain(
        self,
        tau: torch.Tensor,
        selected: torch.Tensor,
        cand_mask: torch.Tensor,
        M: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Unselected real cands. If mask_tau_neg, drop tau^max=-1 (keep 0 and +1)."""
        remain = (cand_mask > 0) & (selected <= 0)
        if self.mask_tau_neg and M is not None:
            remain = remain & (self.tau_max_over_m(tau, M) >= 0)
        return remain

    def comp_feats(
        self,
        h_e: torch.Tensor,
        s_sem: torch.Tensor,
        tau: torch.Tensor,
        selected: torch.Tensor,
        cand_mask: torch.Tensor,
        remain: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gold-free: dup = max sim to already selected; mass = in-TC, like MC, unlike selected."""
        b, n, _ = h_e.shape
        sim = torch.bmm(h_e, h_e.transpose(1, 2))
        in_tc = (tau == 1.0).any(dim=-1).to(h_e.dtype)
        sel_mask = selected > 0
        eye = torch.eye(n, device=h_e.device, dtype=torch.bool).unsqueeze(0)
        key_ok = sel_mask.unsqueeze(1).expand(b, n, n) & ~eye
        neg = torch.finfo(sim.dtype).min
        dup = sim.masked_fill(~key_ok, neg).max(dim=-1).values
        has_sel = sel_mask.any(dim=-1, keepdim=True)
        dup = torch.where(has_sel, dup, torch.zeros_like(dup))
        dup = torch.where(torch.isfinite(dup), dup, torch.zeros_like(dup))
        if remain is None:
            remain = (cand_mask > 0) & (selected <= 0)
        not_dup = 1.0 - dup.clamp(0, 1)
        mass_j = s_sem.clamp(min=0) * in_tc * not_dup * remain.to(h_e.dtype)
        has_remain = remain.any(dim=-1)
        cnt = remain.to(h_e.dtype).sum(dim=-1).clamp(min=1.0)
        rem_mean = mass_j.sum(dim=-1) / cnt
        rem_max = mass_j.max(dim=-1).values
        rem_mean = torch.where(has_remain, rem_mean, torch.zeros_like(rem_mean))
        rem_max = torch.where(has_remain, rem_max, torch.zeros_like(rem_max))
        return dup, mass_j, rem_mean, rem_max

    def sel_logits(
        self,
        s_sem: torch.Tensor,
        tau: torch.Tensor,
        m_time: torch.Tensor,
        g_t: torch.Tensor,
        m_t: torch.Tensor,
        h_ctx: torch.Tensor,
        dup: torch.Tensor,
        mass_j: torch.Tensor,
    ) -> torch.Tensor:
        b, n = s_sem.shape
        g = g_t.unsqueeze(1).expand(-1, n, -1)
        mt = m_t.unsqueeze(1).expand(-1, n, -1)
        x = torch.cat(
            [
                s_sem.unsqueeze(-1),
                tau,
                m_time.unsqueeze(-1),
                g,
                mt,
                h_ctx,
                dup.unsqueeze(-1),
                mass_j.unsqueeze(-1),
            ],
            dim=-1,
        )
        return self.mlp_sel(x).squeeze(-1)

    def stop_logit(
        self,
        g_t: torch.Tensor,
        m_t: torch.Tensor,
        b_norm: torch.Tensor,
        p_sel: torch.Tensor,
        remain: torch.Tensor,
        rem_mean: torch.Tensor,
        rem_max: torch.Tensor,
    ) -> torch.Tensor:
        denom = remain.sum(dim=-1).clamp(min=0.0)
        empty = denom <= 0
        p_mask = p_sel.masked_fill(remain <= 0, 0.0)
        mean_p = p_mask.sum(dim=-1) / denom.clamp(min=1.0)
        max_p = p_sel.masked_fill(remain <= 0, -1e9).max(dim=-1).values
        mean_p = torch.where(empty, torch.zeros_like(mean_p), mean_p)
        max_p = torch.where(empty, torch.zeros_like(max_p), max_p)
        x = torch.cat(
            [
                g_t,
                m_t,
                b_norm.unsqueeze(-1),
                max_p.unsqueeze(-1),
                mean_p.unsqueeze(-1),
                rem_mean.unsqueeze(-1),
                rem_max.unsqueeze(-1),
            ],
            dim=-1,
        )
        return self.mlp_stop(x).squeeze(-1)

    def action_logits(
        self,
        h_e: torch.Tensor,
        s_sem: torch.Tensor,
        tau: torch.Tensor,
        m_time: torch.Tensor,
        g_t: torch.Tensor,
        m_t: torch.Tensor,
        selected: torch.Tensor,
        cand_mask: torch.Tensor,
        b_left: torch.Tensor,
        k: torch.Tensor,
        h_ctx: torch.Tensor,
        M: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.zero_time_feats:
            tau = torch.zeros_like(tau)
            m_time = torch.zeros_like(m_time)
            m_t = torch.zeros_like(m_t)
        remain = self.legal_remain(tau, selected, cand_mask, M)
        dup, mass_j, rem_mean, rem_max = self.comp_feats(
            h_e, s_sem, tau, selected, cand_mask, remain=remain
        )
        logits_j = self.sel_logits(s_sem, tau, m_time, g_t, m_t, h_ctx, dup, mass_j)
        p_j = torch.sigmoid(logits_j)
        b_norm = b_left / k.clamp(min=1.0)
        logit_stop = self.stop_logit(g_t, m_t, b_norm, p_j, remain.float(), rem_mean, rem_max)
        logits = torch.cat([logit_stop.unsqueeze(-1), logits_j], dim=-1)
        mask = torch.ones_like(logits, dtype=torch.bool)
        mask[:, 1:] = remain
        mask[:, 0] = True
        no_budget = b_left <= 0
        mask[no_budget, 1:] = False
        logits = logits.masked_fill(~mask, -1e9)
        return logits, p_j

    def empty_sel_logits(
        self,
        h_e: torch.Tensor,
        s_sem: torch.Tensor,
        tau: torch.Tensor,
        m_time: torch.Tensor,
        cand_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.zero_time_feats:
            tau = torch.zeros_like(tau)
            m_time = torch.zeros_like(m_time)
        b, n = s_sem.shape
        g0 = s_sem.new_zeros(b, self.enc_dim)
        m0 = tau.new_zeros(b, self.m_max)
        selected = s_sem.new_zeros(b, n)
        h_ctx = self.cand_context(h_e, s_sem, tau, m_time, cand_mask)
        dup, mass_j, _, _ = self.comp_feats(h_e, s_sem, tau, selected, cand_mask)
        return self.sel_logits(s_sem, tau, m_time, g0, m0, h_ctx, dup, mass_j)

    def stage0_loss(
        self,
        h_e: torch.Tensor,
        s_sem: torch.Tensor,
        tau: torch.Tensor,
        m_time: torch.Tensor,
        y: torch.Tensor,
        cand_mask: torch.Tensor,
        w_plus: float,
        w_minus: float,
    ) -> torch.Tensor:
        logits = self.empty_sel_logits(h_e, s_sem, tau, m_time, cand_mask)
        p = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)
        w = y * w_plus + (1.0 - y) * w_minus
        nll = -(y * torch.log(p) + (1.0 - y) * torch.log(1.0 - p))
        nll = nll * w * cand_mask
        denom = cand_mask.sum().clamp(min=1.0)
        return nll.sum() / denom

    @torch.no_grad()
    def _sample_actions(self, probs: torch.Tensor, greedy: bool) -> torch.Tensor:
        if greedy:
            return probs.argmax(dim=-1)
        return torch.distributions.Categorical(probs=probs).sample()

    def rollout(
        self,
        batch: dict,
        greedy: bool = False,
        t_max: int = 11,
    ) -> Dict[str, object]:
        h_e, s_sem = self.encode_batch(batch["mc"], batch["texts"])
        tau = batch["tau"]
        m_time = batch["m_time"]
        if self.zero_time_feats:
            tau = torch.zeros_like(tau)
            m_time = torch.zeros_like(m_time)
        cand_mask = batch["cand_mask"]
        k = batch["k"]
        m_seg = batch["M"]
        bsz, n, d = h_e.shape
        h_ctx = self.cand_context(h_e, s_sem, tau, m_time, cand_mask)
        selected = torch.zeros(bsz, n, device=h_e.device)
        g_t = h_e.new_zeros(bsz, d)
        m_t = tau.new_zeros(bsz, self.m_max)
        b_left = k.clone()
        done = torch.zeros(bsz, dtype=torch.bool, device=h_e.device)
        logp_sum = h_e.new_zeros(bsz)
        ent_sum = h_e.new_zeros(bsz)
        order: List[List[int]] = [[] for _ in range(bsz)]

        for _ in range(int(t_max)):
            if bool(done.all()):
                break
            logits, _ = self.action_logits(
                h_e, s_sem, tau, m_time, g_t, m_t, selected, cand_mask, b_left, k, h_ctx, m_seg
            )
            logits = logits - logits.logsumexp(dim=-1, keepdim=True)
            probs = logits.exp()
            dist = torch.distributions.Categorical(probs=probs)
            if greedy:
                act = probs.argmax(dim=-1)
            else:
                act = dist.sample()
            logp = dist.log_prob(act)
            ent = dist.entropy()
            active = ~done
            logp_sum = logp_sum + logp * active.float()
            ent_sum = ent_sum + ent * active.float()

            is_stop = act == 0
            pick = act - 1
            valid_pick = active & (~is_stop) & (pick >= 0) & (pick < n)
            if valid_pick.any():
                b_idx = valid_pick.nonzero(as_tuple=True)[0]
                j_idx = pick[valid_pick]
                ones = torch.ones(b_idx.shape[0], device=h_e.device, dtype=selected.dtype)
                selected = selected.index_put((b_idx, j_idx), ones)
                for i, j in zip(b_idx.tolist(), j_idx.tolist()):
                    order[i].append(j)
                b_left = torch.where(valid_pick, b_left - 1, b_left).detach()
                cnt = selected.sum(dim=-1, keepdim=True).clamp(min=1.0)
                g_new = (h_e * selected.unsqueeze(-1)).sum(dim=1) / cnt
                has = selected.sum(dim=-1, keepdim=True) > 0
                g_t = torch.where(has, g_new, torch.zeros_like(g_new)).detach()
                m_t = self._update_m(m_t, tau, selected, batch["M"]).detach()
                selected = selected.detach()
            done = done | is_stop | (b_left <= 0)
            remain = self.legal_remain(tau, selected, cand_mask, m_seg)
            done = done | (remain.sum(dim=-1) <= 0)

        selected_ids = []
        for i, ids in enumerate(batch["ids"]):
            selected_ids.append([ids[j] for j in order[i] if ids[j]])
        return {
            "logp": logp_sum,
            "entropy": ent_sum,
            "selected_ids": selected_ids,
            "order": order,
        }

    def _update_m(
        self, m_t: torch.Tensor, tau: torch.Tensor, selected: torch.Tensor, M: torch.Tensor
    ) -> torch.Tensor:
        """After any selection: +1 if some selected cand has tau=+1 on that segment, else -1; pad stays 0."""
        hit = ((tau == 1.0) & selected.unsqueeze(-1).bool()).any(dim=1)
        m_new = torch.where(hit, torch.ones_like(m_t), -torch.ones_like(m_t))
        idx = torch.arange(self.m_max, device=m_t.device).unsqueeze(0)
        real = idx < M.unsqueeze(-1)
        has = selected.sum(dim=-1, keepdim=True) > 0
        out = torch.where(real, m_new, torch.zeros_like(m_t))
        return torch.where(has, out, torch.zeros_like(m_t))
