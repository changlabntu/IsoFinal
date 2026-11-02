"""VectorQuantizer2 + optional cosine-normalized codes and dead-code restart.

Both features attack codebook collapse (a few codes doing all the work while
the rest stay dead) — the failure measured on the SA635 MSclean run, where
the coarse scales used only ~14-36 of 256 codes (perplexity 14-36).

`VectorQuantizerVQR(normalize=False, restart=False)` is behaviourally identical
to the parent `taming ...VectorQuantizer2` (only two extra buffers), so it is a
safe drop-in. Both features are OPT-IN — models keep instantiating the parent
directly unless a flag is set, so existing runs are byte-for-byte unaffected.

- normalize: L2-normalize encoder outputs and codes, so assignment is by cosine
  similarity on the unit sphere. Keeps a few codes from dominating by magnitude;
  with a low embed_dim this is the main utilization win (ViT-VQGAN recipe).
- restart: track per-code EMA usage; every `restart_every` steps, reinit codes
  whose usage fell below `restart_thresh` to random current encoder outputs.
  Dead codes get no gradient and can never recover on their own — this revives
  them. Fires only in real training (self.training AND grad enabled), so
  MC-dropout inference never mutates the codebook.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from taming.modules.vqvae.quantize import VectorQuantizer2


class VectorQuantizerVQR(VectorQuantizer2):
    def __init__(self, n_e, e_dim, beta, remap=None, unknown_index="random",
                 sane_index_shape=False, legacy=True,
                 normalize=False, restart=False, restart_every=200,
                 restart_thresh=1.0, usage_decay=0.99):
        super().__init__(n_e, e_dim, beta, remap, unknown_index,
                         sane_index_shape, legacy)
        self.normalize = normalize
        self.restart = restart
        self.restart_every = restart_every
        self.restart_thresh = restart_thresh
        self.usage_decay = usage_decay
        if restart and remap is not None:
            raise NotImplementedError('dead-code restart not supported with --remap')
        # EMA usage counter (for restart) + a step counter, both checkpointed
        self.register_buffer('cluster_size', torch.zeros(n_e))
        self.register_buffer('vqr_step', torch.zeros(1, dtype=torch.long))

    def _codes(self):
        """Codebook as used for lookup/decode — unit-normalized when enabled."""
        w = self.embedding.weight
        return F.normalize(w, dim=1) if self.normalize else w

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp == 1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits is False, "Only for interface compatible with Gumbel"
        assert return_logits is False, "Only for interface compatible with Gumbel"

        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flat = z.view(-1, self.e_dim)
        zf = F.normalize(z_flat, dim=1) if self.normalize else z_flat
        codes = self._codes()

        # (zf - e)^2 = zf^2 + e^2 - 2 zf.e   (== 2 - 2 cos when both normalized)
        d = (zf.pow(2).sum(1, keepdim=True)
             + codes.pow(2).sum(1)
             - 2 * torch.einsum('bd,dn->bn', zf, codes.t()))
        idx = torch.argmin(d, dim=1)
        z_q = F.embedding(idx, codes).view(z.shape)

        # perplexity = effective codebook size this batch (parent returns None here)
        oh = F.one_hot(idx, self.n_e).type(zf.dtype)
        probs = oh.mean(0)
        perplexity = torch.exp(-(probs * (probs + 1e-10).log()).sum())

        # usage tracking + dead-code restart — real training only, never inference
        if self.restart and self.training and torch.is_grad_enabled():
            with torch.no_grad():
                self.cluster_size.mul_(self.usage_decay).add_(
                    oh.sum(0), alpha=1 - self.usage_decay)
                self.vqr_step += 1
                if int(self.vqr_step.item()) % self.restart_every == 0:
                    self._restart_dead(z_flat)

        if not self.legacy:
            loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + \
                   torch.mean((z_q - z.detach()) ** 2)
        else:
            loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * \
                   torch.mean((z_q - z.detach()) ** 2)

        z_q = z + (z_q - z).detach()                      # straight-through
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()

        min_encoding_indices = idx
        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0], -1)
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1, 1)
        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(
                z_q.shape[0], z_q.shape[2], z_q.shape[3])

        return z_q, loss, (perplexity, None, min_encoding_indices)

    def _restart_dead(self, z_flat):
        dead = self.cluster_size < self.restart_thresh
        n_dead = int(dead.sum())
        if n_dead == 0:
            return
        pick = torch.randint(0, z_flat.shape[0], (n_dead,), device=z_flat.device)
        samples = z_flat[pick]                            # raw encoder outputs
        self.embedding.weight.data[dead] = samples        # store un-normalized
        self.cluster_size[dead] = self.restart_thresh     # don't re-restart at once

    def get_codebook_entry(self, indices, shape):
        # must match forward's z_q representation (normalized codes when enabled)
        if self.remap is not None:
            indices = indices.reshape(shape[0], -1)
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(-1)
        z_q = F.embedding(indices, self._codes())
        if shape is not None:
            z_q = z_q.view(shape).permute(0, 3, 1, 2).contiguous()
        return z_q
