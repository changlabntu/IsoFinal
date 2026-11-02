"""Engine — model-agnostic wrapper around the staged generation_test interface.

Any model exposing generation_test(x, method='full'|'encode'|'decode'|
'reconstruction') and latents_from_indices(indices) works (models/MSclean.py is
the reference implementation; models/vqclean.py implements the same contract
with single-scale, length-1 lists).

Contract inherited from generation_test:
- Volumes are (B, C, Y, X, Z), already normalized, at the model's expected Z
  resolution (no dsp/usp/cropz prep here).
- scale_latents is a LIST of per-scale (B, z_channels, H, W, Z) tensors;
  indices a LIST of (B, Z, hk, wk) int64 tensors (the Zarr-storable form).
- Device/dtype are the caller's job: inputs must live on the model's device.
"""

import os

import torch

from inference import registry
from inference.load import REPO_ROOT, load_model, resolve_checkpoint_dir
from inference.normalize import invert_gamma, normalize
from utils.model_utils import read_json_to_args

# Old checkpoints' source snapshots predate generation_test, so for classes with
# a validated current implementation we load the repo's model file instead of
# the snapshot. vqcleanM0aMSskipE -> MSclean is the validated refactor pair
# (same component names; A/B'd to within GPU run-to-run noise, ~7e-4 — see
# test/README.md). Extend this map only after the same A/B validation.
_MODEL_FILE_MAP = {
    'MSclean': 'models/MSclean.py',
    'vqcleanM0aMSskipE': 'models/MSclean.py',
    'vqclean': 'models/vqclean.py',
}


class Engine:
    """A loaded run plus the four staged inference methods."""

    def __init__(self, gan, cfg, spec=None, train_mode=True):
        self.gan = gan   # on device; module-level flag stays eval
        self.cfg = cfg   # the run's config.json as a Namespace
        # Effective normalization: registry spec wins over the run's config.
        spec = spec or {}

        def pick(field, default):
            for v in (spec.get(field), getattr(cfg, field, None)):
                if v is not None:
                    return v
            return default

        self.spec = dict(nm=pick('nm', '00'), gamma=pick('gamma', 0.25),
                         gamma_lo=pick('gamma_lo', -1.0))
        if not hasattr(gan, 'generation_test'):
            raise TypeError(
                f"{type(gan).__name__} has no generation_test — the checkpoint snapshot "
                "predates the staged interface. Pass model_file= pointing at a current "
                "models/*.py with the same component names (see _MODEL_FILE_MAP)."
            )

        # Same convention as test/inference.py: these runs train with batch-stat
        # BN (norm 'batch') and MC dropout, so by default the generator
        # components run in .train() — batch statistics + live dropout (each
        # net_g pass is an MC draw). The 2D VQ stack (encoder/quantizers/
        # decoder) is GroupNorm with dropout 0, so encode() and the codec stay
        # deterministic either way. The GAN module itself stays eval, which is
        # what generation_test's eval-only assert checks. train_mode=False
        # gives fully deterministic running-stat inference.
        if train_mode:
            for name in gan.netg_names:
                getattr(gan, name).train()
        self.train_mode = train_mode

    @classmethod
    def from_checkpoint(cls, path, epoch=None, device=None, model_file=None,
                        train_mode=True):
        """Load a run into an Engine.

        model_file=None auto-resolves known model classes to their current repo
        implementation via _MODEL_FILE_MAP; unknown classes fall back to the
        run's own snapshot (which must then implement generation_test itself).
        train_mode=True (default, the test/ convention) runs the generator
        components with batch-stat BN + MC dropout; False = deterministic.
        """
        if model_file is None:
            ckpt_dir = resolve_checkpoint_dir(path)
            args = read_json_to_args(os.path.join(ckpt_dir, 'config.json'))
            mapped = _MODEL_FILE_MAP.get(args.models)
            if mapped is not None:
                model_file = os.path.join(REPO_ROOT, mapped)
        gan, cfg = load_model(path, epoch=epoch, device=device, model_file=model_file)
        return cls(gan, cfg, train_mode=train_mode)

    @classmethod
    def from_registry(cls, name, epoch=None, device=None, train_mode=True):
        """Load a model registered in inference/registry.py by name.

        The spec pins checkpoint, model_file, epoch, and normalization; the
        epoch/device/train_mode arguments override the spec when given.
        """
        spec = registry.get(name)
        model_file = spec.get('model_file')
        if model_file is not None and not os.path.isabs(model_file):
            model_file = os.path.join(REPO_ROOT, model_file)
        gan, cfg = load_model(spec['checkpoint'],
                              epoch=epoch if epoch is not None else spec.get('epoch'),
                              device=device, model_file=model_file)
        # a registry value that contradicts the run's own config is probably a
        # stale copy-paste — normalization must match training exactly
        for field in ('nm', 'gamma', 'gamma_lo'):
            cfg_v = getattr(cfg, field, None)
            if field in spec and cfg_v is not None and spec[field] != cfg_v:
                print(f"WARNING: registry '{name}' {field}={spec[field]} differs from "
                      f"the run's config.json {field}={cfg_v} — using the registry value")
        return cls(gan, cfg, spec=spec, train_mode=train_mode)

    @property
    def device(self):
        return next(self.gan.parameters()).device

    @torch.inference_mode()
    def encode(self, x):
        """(B, C, Y, X, Z) volume -> (scale_latents, indices)."""
        return self.gan.generation_test(x, method='encode')

    @torch.inference_mode()
    def decode(self, scale_latents):
        """scale_latents list -> (B, 1, Y, X, Z*uprate) isotropic volume (Y, X unchanged)."""
        return self.gan.generation_test(scale_latents, method='decode')

    @torch.inference_mode()
    def decode_ms(self, scale_latents, stop_at=None):
        """scale_latents list -> net_g's multi-scale output dict ('out0' full
        res, 'out128' 1/2, 'out64' 1/4 — one shared decoder pass). MS netGs
        only. stop_at='out64'|'out128' early-exits after that head (cheaper);
        the dict then holds only the heads computed so far."""
        return self.gan.generation_test(scale_latents, method='decode_ms',
                                        stop_at=stop_at)

    @torch.inference_mode()
    def full(self, x):
        """(B, C, Y, X, Z) volume -> (B, 1, Y, X, Z*uprate) isotropic volume (Y, X unchanged)."""
        return self.gan.generation_test(x, method='full')

    @torch.inference_mode()
    def reconstruction(self, scale_latents):
        """scale_latents list -> (B, out_ch, Y, X, Z) slice-wise VQ-head recon."""
        return self.gan.generation_test(scale_latents, method='reconstruction')

    @torch.inference_mode()
    def latents_from_indices(self, indices):
        """Stored indices list -> scale_latents list (no encoder run)."""
        return self.gan.latents_from_indices(indices)

    # -- normalization convenience, driven by the effective spec ------------

    def normalize(self, vol, norm_stats=None, key=None):
        """Apply the model's training normalization (spec nm/gamma/gamma_lo)
        to a raw volume (numpy or torch, any shape)."""
        return normalize(vol, self.spec['nm'], gamma=self.spec['gamma'],
                         gamma_lo=self.spec['gamma_lo'],
                         norm_stats=norm_stats, key=key)

    def denormalize(self, vol, gamma=None, gamma_lo=None):
        """Invert nm='11g' outputs back to the pre-gamma [-1, 1] scale
        (no-op for other nm modes). gamma/gamma_lo re-tone the inversion only
        (the model input keeps the trained transform)."""
        if self.spec['nm'] != '11g':
            return vol
        return invert_gamma(vol,
                            self.spec['gamma'] if gamma is None else gamma,
                            self.spec['gamma_lo'] if gamma_lo is None else gamma_lo)
