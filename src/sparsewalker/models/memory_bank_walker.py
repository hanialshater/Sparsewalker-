import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .sparsewalker import SparseWalker


class SparseWalkerMemoryBank(SparseWalker):
    """Sparse Walker with a dynamic bank of addressable historical sparse states.

    The recurrent state remains K sparse concepts. During the first graph hop,
    the current state routes to the top few states in a small per-user memory
    bank, imports their concepts, and then continues the normal concept walk.

    Memory writes are discrete and detached: a new state is stored only when it
    is sufficiently novel relative to existing bank keys and sufficiently far
    from the previous write. This keeps backward bounded while testing the core
    hypothesis that long histories need sparse addressable access to old states.
    """

    def __init__(
        self,
        n_items,
        max_len,
        d=64,
        layers=2,
        side=256,
        h=16,
        active=8,
        top_side=2,
        degree=4,
        fresh_weight=.25,
        bank_size=16,
        memory_topk=2,
        novelty_threshold=.85,
        min_write_gap=8,
        initial_memory_share=.25,
    ):
        super().__init__(
            n_items, max_len, d=d, layers=layers, side=side, h=h,
            active=active, top_side=top_side, degree=degree,
            fresh_weight=fresh_weight,
        )
        self.bank_size = int(bank_size)
        self.memory_topk = int(memory_topk)
        self.novelty_threshold = float(novelty_threshold)
        self.min_write_gap = int(min_write_gap)
        if self.bank_size <= 0:
            raise ValueError("bank_size must be positive")
        if not (1 <= self.memory_topk <= self.bank_size):
            raise ValueError("memory_topk must be in [1, bank_size]")
        if self.min_write_gap < 1:
            raise ValueError("min_write_gap must be positive")

        self.memory_q = nn.Linear(d, d, bias=False)
        p = float(initial_memory_share)
        if not 0.0 < p < 1.0:
            raise ValueError("initial_memory_share must be in (0,1)")
        self.memory_share_logit = nn.Parameter(
            torch.tensor(math.log(p / (1.0 - p)))
        )
        self.memory_enabled = True
        self._diag_enabled = False
        self.reset_diagnostics()

    def reset_diagnostics(self):
        self._diag = {
            "steps": 0.0,
            "steps_with_bank": 0.0,
            "selected_states": 0.0,
            "selected_similarity_sum": 0.0,
            "selected_age_sum": 0.0,
            "selected_age_ge32": 0.0,
            "selected_age_ge64": 0.0,
            "writes": 0.0,
            "write_similarity_sum": 0.0,
            "write_similarity_count": 0.0,
            "final_occupancy_sum": 0.0,
            "final_occupancy_users": 0.0,
        }

    def enable_diagnostics(self, enabled=True):
        self._diag_enabled = bool(enabled)
        if enabled:
            self.reset_diagnostics()

    def diagnostics_summary(self):
        d = self._diag
        selected = max(1.0, d["selected_states"])
        writes_with_sim = max(1.0, d["write_similarity_count"])
        users = max(1.0, d["final_occupancy_users"])
        return {
            "steps": int(d["steps"]),
            "memory_available_rate": d["steps_with_bank"] / max(1.0, d["steps"]),
            "selected_states": int(d["selected_states"]),
            "mean_selected_similarity": d["selected_similarity_sum"] / selected,
            "mean_selected_age": d["selected_age_sum"] / selected,
            "retrieval_age_ge32_rate": d["selected_age_ge32"] / selected,
            "retrieval_age_ge64_rate": d["selected_age_ge64"] / selected,
            "writes": int(d["writes"]),
            "mean_similarity_at_novel_write": d["write_similarity_sum"] / writes_with_sim,
            "mean_final_bank_occupancy": d["final_occupancy_sum"] / users,
            "memory_share": float(torch.sigmoid(self.memory_share_logit).detach().cpu()),
        }

    def _state_vector(self, ids, mass):
        return (self.space.value(ids) * mass[..., None]).sum(-2)

    def _read_bank(
        self,
        old_ids,
        old_mass,
        fresh_ids,
        fresh_mass,
        context,
        bank_ids,
        bank_mass,
        bank_key,
        bank_valid,
        bank_time,
        step,
    ):
        if not self.memory_enabled:
            return self._merge(old_ids, old_mass, fresh_ids, fresh_mass), None

        old_vec = self._state_vector(old_ids, old_mass)
        q = F.normalize(self.memory_q(context + self.message_proj(old_vec)), dim=-1)
        score = (bank_key * q[:, None, :]).sum(-1)
        score = score.masked_fill(~bank_valid, -1e4)

        k = self.memory_topk
        topv, topi = score.topk(k, dim=-1)
        top_valid = bank_valid.gather(1, topi)
        safe = topv.masked_fill(~top_valid, -1e4)
        gates = F.softmax(safe, dim=-1) * top_valid.to(safe.dtype)
        gates = gates / (gates.sum(-1, keepdim=True) + 1e-8)

        gather_idx = topi[..., None].expand(-1, -1, self.active)
        rid = bank_ids.gather(1, gather_idx)
        rmass = bank_mass.gather(1, gather_idx)
        rtime = bank_time.gather(1, topi)

        old_budget = 1.0 - self.fresh_weight
        share = torch.sigmoid(self.memory_share_logit)
        has_memory = top_valid.any(-1).to(old_mass.dtype)
        mem_weight = old_budget * share * has_memory
        old_weight = old_budget - mem_weight

        ids = torch.cat(
            [old_ids, fresh_ids, rid.reshape(rid.size(0), -1)], dim=-1
        )
        mass = torch.cat(
            [
                old_weight[:, None] * old_mass,
                self.fresh_weight * fresh_mass,
                (mem_weight[:, None, None] * gates[..., None] * rmass).reshape(rmass.size(0), -1),
            ],
            dim=-1,
        )
        out = self._top(ids, mass)

        if self._diag_enabled:
            with torch.no_grad():
                valid_steps = old_mass.sum(-1) > 0
                self._diag["steps"] += float(valid_steps.sum().item())
                has = top_valid.any(-1) & valid_steps
                self._diag["steps_with_bank"] += float(has.sum().item())
                sv = topv[top_valid & valid_steps[:, None]]
                ages = (step - rtime).clamp_min(0)[top_valid & valid_steps[:, None]]
                self._diag["selected_states"] += float(sv.numel())
                if sv.numel():
                    self._diag["selected_similarity_sum"] += float(sv.sum().item())
                    self._diag["selected_age_sum"] += float(ages.float().sum().item())
                    self._diag["selected_age_ge32"] += float((ages >= 32).sum().item())
                    self._diag["selected_age_ge64"] += float((ages >= 64).sum().item())

        return out, topi

    @torch.no_grad()
    def _write_bank(
        self,
        ids,
        mass,
        act,
        bank_ids,
        bank_mass,
        bank_key,
        bank_valid,
        bank_time,
        bank_last_access,
        last_write_step,
        step,
    ):
        state_key = F.normalize(self._state_vector(ids, mass).detach(), dim=-1)
        sim = (bank_key * state_key[:, None, :]).sum(-1)
        sim = sim.masked_fill(~bank_valid, -1e4)
        any_valid = bank_valid.any(-1)
        max_sim = sim.max(-1).values
        gap_ok = (step - last_write_step) >= self.min_write_gap
        should = act & gap_ok & ((~any_valid) | (max_sim < self.novelty_threshold))

        free = ~bank_valid
        has_free = free.any(-1)
        first_free = free.to(torch.int64).argmax(-1)
        lru = bank_last_access.argmin(-1)
        slot = torch.where(has_free, first_free, lru)

        write_mask = F.one_hot(slot, self.bank_size).bool() & should[:, None]
        wm3 = write_mask[..., None]
        bank_ids = torch.where(wm3, ids.detach()[:, None, :], bank_ids)
        bank_mass = torch.where(wm3, mass.detach()[:, None, :], bank_mass)
        bank_key = torch.where(wm3, state_key[:, None, :], bank_key)
        bank_valid = bank_valid | write_mask
        bank_time = torch.where(write_mask, torch.full_like(bank_time, step), bank_time)
        bank_last_access = torch.where(write_mask, torch.full_like(bank_last_access, step), bank_last_access)
        last_write_step = torch.where(should, torch.full_like(last_write_step, step), last_write_step)

        if self._diag_enabled:
            self._diag["writes"] += float(should.sum().item())
            novel_after_first = should & any_valid
            if novel_after_first.any():
                self._diag["write_similarity_sum"] += float(max_sim[novel_after_first].sum().item())
                self._diag["write_similarity_count"] += float(novel_after_first.sum().item())

        return (
            bank_ids, bank_mass, bank_key, bank_valid, bank_time,
            bank_last_access, last_write_step,
        )

    def _encode_impl(self, seq, return_states):
        B, L = seq.shape
        valid = seq != 0
        item_state = self.item(seq) * math.sqrt(self.d_model)
        fi, fm = self.router(item_state.reshape(B * L, self.d_model), self.space)
        fi = fi.view(B, L, -1)
        fm = fm.view(B, L, -1)

        ids = torch.zeros(B, self.active, dtype=torch.long, device=seq.device)
        mass = torch.zeros(B, self.active, dtype=item_state.dtype, device=seq.device)

        bank_ids = torch.zeros(B, self.bank_size, self.active, dtype=torch.long, device=seq.device)
        bank_mass = torch.zeros(B, self.bank_size, self.active, dtype=item_state.dtype, device=seq.device)
        bank_key = torch.zeros(B, self.bank_size, self.d_model, dtype=item_state.dtype, device=seq.device)
        bank_valid = torch.zeros(B, self.bank_size, dtype=torch.bool, device=seq.device)
        bank_time = torch.zeros(B, self.bank_size, dtype=torch.long, device=seq.device)
        bank_last_access = torch.zeros(B, self.bank_size, dtype=torch.long, device=seq.device)
        last_write_step = torch.full((B,), -self.min_write_gap, dtype=torch.long, device=seq.device)

        outs = []
        ids_hist = [] if return_states else None
        mass_hist = [] if return_states else None
        touched_sources = [] if self.training else None

        for t in range(L):
            step = t + 1
            act = valid[:, t]
            af = act.to(item_state.dtype)[:, None]
            xids = ids
            xmass = mass * af
            fids = fi[:, t]
            fmass = fm[:, t] * af

            (xids, xmass), selected = self._read_bank(
                xids, xmass, fids, fmass, item_state[:, t],
                bank_ids, bank_mass, bank_key, bank_valid, bank_time, step,
            )
            if selected is not None:
                with torch.no_grad():
                    access = F.one_hot(selected, self.bank_size).any(1) & bank_valid
                    bank_last_access = torch.where(
                        access, torch.full_like(bank_last_access, step), bank_last_access
                    )

            if touched_sources is not None:
                touched_sources.append(xids.detach())
            xids, xmass = self.graph(
                xids, xmass, item_state[:, t], self.space, track_touched=False
            )

            for _ in range(1, self.layers_n):
                xids, xmass = self._merge(xids, xmass, fids, fmass)
                if touched_sources is not None:
                    touched_sources.append(xids.detach())
                xids, xmass = self.graph(
                    xids, xmass, item_state[:, t], self.space, track_touched=False
                )

            ids = torch.where(act[:, None], xids, ids)
            mass = torch.where(act[:, None], xmass, mass)

            (
                bank_ids, bank_mass, bank_key, bank_valid, bank_time,
                bank_last_access, last_write_step,
            ) = self._write_bank(
                ids, mass, act, bank_ids, bank_mass, bank_key, bank_valid,
                bank_time, bank_last_access, last_write_step, step,
            )

            msg = self._state_vector(ids, mass)
            h = self.norm(item_state[:, t] + self.message_proj(msg)) * af
            outs.append(h)
            if return_states:
                ids_hist.append(ids)
                mass_hist.append(mass)

        if touched_sources:
            self.graph.mark_touched(torch.cat([x.reshape(-1) for x in touched_sources], 0))

        if self._diag_enabled:
            with torch.no_grad():
                occupancy = bank_valid.sum(-1).float()
                self._diag["final_occupancy_sum"] += float(occupancy.sum().item())
                self._diag["final_occupancy_users"] += float(B)

        H = torch.stack(outs, 1)
        if return_states:
            return H, torch.stack(ids_hist, 1), torch.stack(mass_hist, 1)
        return H
