import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .sparsewalker import SparseWalker


class SparseWalkerTemporalMemory(SparseWalker):
    """Sparse Walker with three sparse temporal landmark memories.

    The base Walker state still has K active concepts and follows the same
    learned concept graph. In addition, a tiny set of frozen-in-time concept
    snapshots can be read before the first graph hop at each event. This gives
    old information a direct temporal shortcut instead of requiring it to
    survive every intermediate merge/top-k transition.

    Landmark writes are detached gradient boundaries. This is intentional:
    without detaching, every saved state is reused by many later time steps and
    creates a large fan-out of long BPTT branches. Forward information still
    travels through the skip, while the memory query/gating and current concept
    values remain trainable.

    The default schedules are intentionally staggered for max_len~=200:
      short  : every 16 events       -> recent landmark
      medium : every 64, offset 16   -> mid-range landmark
      long   : every 256, offset 64  -> one durable old landmark

    With K=8 this stores only 3*K=24 concept ids/masses per user.
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
        memory_periods=(16, 64, 256),
        memory_offsets=(0, 16, 64),
        initial_memory_share=.25,
    ):
        super().__init__(
            n_items,
            max_len,
            d=d,
            layers=layers,
            side=side,
            h=h,
            active=active,
            top_side=top_side,
            degree=degree,
            fresh_weight=fresh_weight,
        )
        if len(memory_periods) != len(memory_offsets):
            raise ValueError("memory_periods and memory_offsets must match")
        if len(memory_periods) == 0:
            raise ValueError("at least one temporal memory is required")
        self.memory_periods = tuple(int(x) for x in memory_periods)
        self.memory_offsets = tuple(int(x) for x in memory_offsets)
        if any(x <= 0 for x in self.memory_periods):
            raise ValueError("memory periods must be positive")
        self.n_memories = len(self.memory_periods)

        self.memory_q = nn.Linear(d, d, bias=False)
        self.memory_bias = nn.Parameter(torch.zeros(self.n_memories))

        p = float(initial_memory_share)
        if not 0.0 < p < 1.0:
            raise ValueError("initial_memory_share must be in (0,1)")
        self.memory_share_logit = nn.Parameter(torch.tensor(math.log(p / (1.0 - p))))

    def _read_memory(self, old_ids, old_mass, fresh_ids, fresh_mass, context,
                     memory_ids, memory_mass, memory_valid):
        """Merge current/fresh state with a query-weighted sparse memory read."""
        # IDs/masses are detached snapshots, but concept values are looked up
        # through the live ConceptSpace so representation learning still flows.
        mem_values = self.space.value(memory_ids)
        mem_vec = (mem_values * memory_mass[..., None]).sum(2)
        q = F.normalize(self.memory_q(context), dim=-1)
        mk = F.normalize(mem_vec, dim=-1)
        score = (mk * q[:, None, :]).sum(-1) + self.memory_bias[None, :]

        valid_f = memory_valid.to(score.dtype)
        shifted = score - score.max(-1, keepdim=True).values
        raw = torch.exp(shifted) * valid_f
        gates = raw / (raw.sum(-1, keepdim=True) + 1e-8)

        old_budget = 1.0 - self.fresh_weight
        share = torch.sigmoid(self.memory_share_logit)
        has_memory = memory_valid.any(-1).to(old_mass.dtype)
        mem_weight = old_budget * share * has_memory
        old_weight = old_budget - mem_weight

        mem_weights = mem_weight[:, None, None] * gates[:, :, None]
        ids = torch.cat(
            [old_ids, fresh_ids, memory_ids.reshape(memory_ids.size(0), -1)],
            dim=-1,
        )
        mass = torch.cat(
            [
                old_weight[:, None] * old_mass,
                self.fresh_weight * fresh_mass,
                (mem_weights * memory_mass).reshape(memory_mass.size(0), -1),
            ],
            dim=-1,
        )
        return self._top(ids, mass)

    def _should_write(self, step, memory_index):
        period = self.memory_periods[memory_index]
        offset = self.memory_offsets[memory_index]
        if step < offset:
            return False
        return (step - offset) % period == 0

    def _encode_impl(self, seq, return_states):
        B, L = seq.shape
        valid = seq != 0
        item_state = self.item(seq) * math.sqrt(self.d_model)
        fi, fm = self.router(item_state.reshape(B * L, self.d_model), self.space)
        fi = fi.view(B, L, -1)
        fm = fm.view(B, L, -1)

        ids = torch.zeros(B, self.active, dtype=torch.long, device=seq.device)
        mass = torch.zeros(B, self.active, dtype=item_state.dtype, device=seq.device)

        memory_ids = [torch.zeros_like(ids) for _ in range(self.n_memories)]
        memory_mass = [torch.zeros_like(mass) for _ in range(self.n_memories)]
        memory_valid = [torch.zeros(B, dtype=torch.bool, device=seq.device)
                        for _ in range(self.n_memories)]

        outs = []
        ids_hist = [] if return_states else None
        mass_hist = [] if return_states else None
        touched_sources = [] if self.training else None

        for t in range(L):
            act = valid[:, t]
            af = act.to(item_state.dtype)[:, None]
            xids = ids
            xmass = mass * af
            fids = fi[:, t]
            fmass = fm[:, t] * af

            mids = torch.stack(memory_ids, 1)
            mmass = torch.stack(memory_mass, 1)
            mvalid = torch.stack(memory_valid, 1)

            xids, xmass = self._read_memory(
                xids, xmass, fids, fmass, item_state[:, t], mids, mmass, mvalid
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

            # Detached writes bound backward complexity while preserving the
            # exact forward landmark state. IDs are discrete already; masses
            # are the important gradient boundary.
            step = t + 1
            for j in range(self.n_memories):
                if self._should_write(step, j):
                    snap_ids = ids.detach()
                    snap_mass = mass.detach()
                    memory_ids[j] = torch.where(act[:, None], snap_ids, memory_ids[j])
                    memory_mass[j] = torch.where(act[:, None], snap_mass, memory_mass[j])
                    memory_valid[j] = memory_valid[j] | act

            msg = (self.space.value(ids) * mass[:, :, None]).sum(1)
            h = self.norm(item_state[:, t] + self.message_proj(msg)) * af
            outs.append(h)
            if return_states:
                ids_hist.append(ids)
                mass_hist.append(mass)

        if touched_sources:
            self.graph.mark_touched(torch.cat([x.reshape(-1) for x in touched_sources], 0))

        H = torch.stack(outs, 1)
        if return_states:
            return H, torch.stack(ids_hist, 1), torch.stack(mass_hist, 1)
        return H
