from models.base import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager

from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
from ldm.modules.diffusionmodules.modelcut import Encoder, Decoder
from ldm.util import instantiate_from_config
import yaml
import numpy as np
from torch.optim.lr_scheduler import LambdaLR
from networks.networks import get_scheduler
import os
from pytorch_msssim import ms_ssim
import tifffile as tiff


# MSclean: vqcleanM0aMSskipE, refactored (same math — verified bitwise against the
# original, forward and backward). Requires --netG ed023emsfpn and num_scales >= 3;
# run with --pyr_detach --adv_ms 0.5.
#
# New vs vqcleanM0aMSskipE:
#   - One shape convention: volumes (B, C, Y, X, Z) <-> slice batches (B*Z, C, Y, X)
#     via vol_to_slices/slices_to_vol; encode/decode/_netg_decode shared by training
#     and inference (no duplicated trunk code).
#   - TRUE batch_size > 1 training: all losses fold B into the slice/plane axes
#     (needs equal native Z within a batch; memory scales ~linearly with B).
#   - Staged stateless inference: generation_test(x, method='full'|'encode'|'decode'|
#     'reconstruction') and latents_from_indices() for the stored-indices
#     compression path.
#
# Inherited from vqcleanM0aMSskip: full-sum trunk (every code gets full decoder depth) +
# zero-init lateral skips of the two finest VQ scales after the out64/out128 taps, so the
# model starts numerically identical to vqcleanM0aMS.
#
# New here:
#   F1 — EMA is ALWAYS ON (--ema_decay, default 0.9999 with warmup): validation and the
#        epoch checkpoints run under EMA weights (on_validation_start/end swap them in and
#        out; training_epoch_end wraps the base checkpoint save the same way), so metrics,
#        GIFs, and saved nets all reflect the averaged generator.
#   F2 — the coarse heads are supervised against REAL data, not just the model's own
#        pooled output: out128/out64 are Z-projected (same skipl1/--l1how as the main l1)
#        and L1-matched to the XY-avg-pooled input (weight --lamb * --lamb_coarse,
#        logged as 'l1c'). Requires uprate divisible by 4.
class GAN(BaseModel):
    def __init__(self, hparams, train_loader, eval_loader, checkpoints):
        BaseModel.__init__(self, hparams, train_loader, eval_loader, checkpoints)

        print('Reading yaml: ' + self.hparams.ldmyaml)
        with open('ldm/' + self.hparams.ldmyaml + '.yaml', "r") as f:
            config = yaml.load(f, Loader=yaml.Loader)
        ddconfig = config['model']['params']["ddconfig"]

        if self.hparams.tc:
            ddconfig['in_channels'] = 2
            ddconfig['out_ch'] = 1
        self.hparams.netG = self.hparams.netG

        self.hparams.final = 'tanh'
        self.net_g, self.net_d = self.set_networks()

        # Separate per-scale discriminators for the coarse (1/2, 1/4) outputs. The patch
        # discriminator is fully convolutional, so it scores slices at any resolution.
        self.net_d_128 = self.set_networks(net='d')
        self.net_d_64 = self.set_networks(net='d')

        # VQGAN components
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.embed_dim = config['model']['params']['embed_dim']
        self.n_embed = config['model']['params']['n_embed']

        # Multi-scale VQ — scale_factors go coarse to fine, matching VAR Algorithm 1
        # e.g. num_scales=3 → scale_factors=[4, 2, 1]
        # num_scales=1 → scale_factors=[1] (identical to original single VQ)
        self.num_scales = getattr(hparams, 'num_scales', 1)
        if self.num_scales < 3:
            raise ValueError('MSclean needs --num_scales >= 3 (two of the scales '
                             'are laterally injected); got %d' % self.num_scales)
        self.scale_factors = [2 ** (self.num_scales - 1 - i) for i in range(self.num_scales)]
        print('scale_factors: ' + str(self.scale_factors))

        # Per-scale quant_conv: z_channels → embed_dim (applied at downsampled resolution)
        self.quant_convs = nn.ModuleList([
            nn.Conv2d(ddconfig["z_channels"], self.embed_dim, 1)
            for _ in range(self.num_scales)
        ])

        # Per-scale post_quant_conv: embed_dim → z_channels
        # Applied AFTER upsampling back to full resolution, matching VAR ϕk
        self.post_quant_convs = nn.ModuleList([
            nn.Conv2d(self.embed_dim, ddconfig["z_channels"], 1)
            for _ in range(self.num_scales)
        ])

        # Quantizers — separate codebook per scale by default
        # Set --shared_codebook to use a single codebook across all scales (VAR paper)
        self.shared_codebook = getattr(hparams, 'shared_codebook', False)

        # Per-scale codebook sizes (coarse->fine). Default: the yaml n_embed for
        # every scale, so an unset flag reproduces the old uniform allocation.
        scales_arg = getattr(hparams, 'n_embed_scales', None)
        if scales_arg:
            sizes = [int(s) for s in str(scales_arg).split(',')]
            if len(sizes) != self.num_scales:
                raise ValueError('--n_embed_scales needs %d comma-separated sizes, got %d'
                                 % (self.num_scales, len(sizes)))
            if self.shared_codebook:
                raise ValueError('--n_embed_scales requires separate codebooks '
                                 '(drop --shared_codebook)')
        else:
            sizes = [self.n_embed] * self.num_scales

        # Collapse mitigations are opt-in: without --vq_normalize/--vq_restart the
        # plain taming VectorQuantizer is instantiated, byte-identical to before.
        vq_normalize = getattr(hparams, 'vq_normalize', False)
        vq_restart = getattr(hparams, 'vq_restart', False)

        def _make_quantizer(n_e):
            common = dict(remap=getattr(hparams, 'remap', None),
                          sane_index_shape=getattr(hparams, 'sane_index_shape', False))
            if vq_normalize or vq_restart:
                from models.quantize import VectorQuantizerVQR
                return VectorQuantizerVQR(
                    n_e, self.embed_dim, beta=0.25,
                    normalize=vq_normalize, restart=vq_restart,
                    restart_every=getattr(hparams, 'vq_restart_every', 200),
                    restart_thresh=getattr(hparams, 'vq_restart_thresh', 1.0),
                    **common)
            return VectorQuantizer(n_e, self.embed_dim, beta=0.25, **common)

        if self.shared_codebook:
            shared_q = _make_quantizer(self.n_embed)
            self.quantizers = nn.ModuleList([shared_q for _ in range(self.num_scales)])
        else:
            self.quantizers = nn.ModuleList([_make_quantizer(s) for s in sizes])

        # Adapters projecting the injected VQ scales (z_channels, latent res) to the
        # net_g decode-stage channel counts (nf follows set_networks' nf=hparams.ngf).
        # net_g resizes them spatially at the injection point, so 1x1 convs suffice here.
        self.inject_convs = nn.ModuleList([
            nn.Conv3d(ddconfig["z_channels"], 4 * hparams.ngf, 1),   # 'pre_up2' (4*nf stage)
            nn.Conv3d(ddconfig["z_channels"], 2 * hparams.ngf, 1),   # 'pre_up1' (2*nf stage)
        ])
        # Zero-init (ControlNet-style): skips contribute nothing at step 0, so the model
        # starts numerically identical to vqcleanM0aMS and can only learn helpful skips.
        for m in self.inject_convs:
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

        # Initialize loss
        self.loss = instantiate_from_config(config['model']['params']["lossconfig"])
        self.discriminator = self.loss.discriminator

        # EMA support
        # MSskipE (F1): EMA is always on — validation and epoch checkpoints use EMA weights.
        self.use_ema = True
        from ldm.modules.ema import LitEma
        self.model_ema = LitEma(self, decay=getattr(hparams, 'ema_decay', 0.9999))
        print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.netg_names = {
            'encoder': 'encoder',
            'decoder': 'decoder',
            'quantizers': 'quantizers',
            'quant_convs': 'quant_convs',
            'post_quant_convs': 'post_quant_convs',
            'inject_convs': 'inject_convs',
            'net_g': 'net_g'
        }
        self.netd_names = {
            'discriminator': 'discriminator',
            'net_d': 'net_d',
            'net_d_128': 'net_d_128',
            'net_d_64': 'net_d_64',
        }

        self.configure_optimizers()

        self.upsample = torch.nn.Upsample(
            size=(hparams.cropsize, hparams.cropsize, hparams.cropsize), mode='trilinear'
        )
        self.uprate = (hparams.cropsize // hparams.cropz) * hparams.dsp / hparams.usp
        self.uprate = int(self.uprate)
        if self.hparams.lamb_coarse > 0 and self.uprate % 4 != 0:
            raise ValueError('MSclean coarse supervision (--lamb_coarse > 0) needs '
                             'uprate divisible by 4; got %d' % self.uprate)
        print('uprate: ' + str(self.uprate))
        print('num_scales: ' + str(self.num_scales))

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("VQGAN")
        parser.add_argument("--skipl1", type=int, default=4)
        parser.add_argument("--l1how", type=str, default='dsp')
        parser.add_argument("--dsp", type=int, default=1, help='extra downsample rate')
        parser.add_argument("--usp", type=float, default=1.0, help='extra upsample rate')
        parser.add_argument("--downbranch", type=int, default=1)
        parser.add_argument("--resizebranch", type=float, default=1.0)
        parser.add_argument('--lbm_ms_ssim', type=float, default=0, help='weight for ms_ssim loss')
        # Multi-scale (progressive) output supervision
        parser.add_argument("--lamb_pyr", type=float, default=1.0,
                            help='weight for pyramid consistency loss on the coarse (1/2, 1/4) output heads')
        parser.add_argument("--pyr_detach", action='store_true', default=False,
                            help='detach the avg-pooled full-res target so pyramid consistency only trains the coarse heads (distillation)')
        parser.add_argument("--adv_ms", type=float, default=1.0,
                            help='multiplier on the 1/2 and 1/4 adversarial loss (on top of --adv); adv_ms=0 skips the coarse discriminators entirely')
        # VQ specific arguments
        parser.add_argument("--ldmyaml", type=str, default='vqgan')
        parser.add_argument("--use_ema", action='store_true',
                            help='(ignored — EMA is always on in this model)')
        parser.add_argument("--ema_decay", type=float, default=0.9999,
                            help='EMA decay (LitEma warms up from 0 via its num_updates schedule)')
        parser.add_argument("--lamb_coarse", type=float, default=1.0,
                            help='weight of the real-data coarse L1 (out128/out64 Z-projections vs '
                                 'XY-pooled input), multiplied on top of --lamb; 0 disables')
        parser.add_argument("--remap", type=str, default=None, help='remap indices')
        parser.add_argument("--sane_index_shape", action='store_true', help='return indices as bhw')
        parser.add_argument("--lr_g_factor", type=float, default=1.0, help='learning rate factor for generator')
        parser.add_argument("--num_scales", type=int, default=4,
                            help='number of VQ scales coarse-to-fine (this model needs >= 3: '
                                 'coarse trunk + two injected scales)')
        parser.add_argument("--shared_codebook", action='store_true',
                            help='share a single codebook across all scales (VAR paper setting)')
        # Codebook-collapse mitigations (all opt-in; unset = identical to before).
        parser.add_argument("--n_embed_scales", type=str, default=None,
                            help='comma-separated per-scale codebook sizes coarse->fine, '
                                 'e.g. "64,64,256,512"; default: the ldm yaml n_embed for '
                                 'every scale. Requires separate codebooks.')
        parser.add_argument("--vq_normalize", action='store_true',
                            help='cosine VQ: L2-normalize codes and encoder outputs (improves '
                                 'codebook utilization; best with a low embed_dim)')
        parser.add_argument("--vq_restart", action='store_true',
                            help='dead-code restart: periodically reinit unused codes to '
                                 'current encoder outputs')
        parser.add_argument("--vq_restart_every", type=int, default=200,
                            help='steps between dead-code restarts (with --vq_restart)')
        parser.add_argument("--vq_restart_thresh", type=float, default=1.0,
                            help='EMA-usage below which a code is dead (with --vq_restart)')
        return parent_parser

    # ------------------------------------------------------------------
    # Shape conventions
    # ------------------------------------------------------------------
    # Volumes are (B, C, Y, X, Z) — the repo-wide layout (dim4 = Z, the
    # anisotropic axis). The 2D VQ stack (encoder / quantizers / decoder)
    # runs on XY-slice batches (B*Z, C, Y, X): Z is folded into the batch
    # dimension. These two helpers are the ONLY volume<->slice conversions.

    @staticmethod
    def vol_to_slices(v):
        """(B, C, Y, X, Z) volume -> (B*Z, C, Y, X) XY-slice batch."""
        B, C, Y, X, Z = v.shape
        return v.permute(0, 4, 1, 2, 3).reshape(B * Z, C, Y, X)

    @staticmethod
    def slices_to_vol(s, batch_size=1):
        """(B*Z, C, H, W) slice batch -> (B, C, H, W, Z) volume."""
        BZ, C, H, W = s.shape
        return s.reshape(batch_size, BZ // batch_size, C, H, W).permute(0, 2, 3, 4, 1)

    def encode(self, x):
        """
        Residual multi-scale VQ encoding following VAR Algorithm 1.

        In:  x — (N, C, Y, X) slice batch (N = B*Z XY slices of a volume).
        Out: scale_latents — list of num_scales per-scale contributions, each
                             (N, z_channels, H, W) at the full latent resolution;
                             sum(scale_latents) is the combined latent that
                             decode() / the net_g trunk consume.
             emb_loss      — codebook loss, averaged over scales.
             indices       — per-scale codebook indices (store these in Zarr).

        Per scale k (coarse to fine):
          1. downsample the residual to Hk × Wk
          2. quant_convs[k]: z_channels → embed_dim
          3. quantizers[k]: quantize → code_k (embed_dim space)
          4. upsample code_k back to full H × W
          5. post_quant_convs[k]: embed_dim → z_channels (VAR ϕk) → latent_k
          6. residual = residual − latent_k
        """
        h, _, _ = self.encoder(x)
        H, W = h.shape[-2:]

        residual = h          # z_channels space, full H × W
        scale_latents = []
        emb_losses = []
        indices = []
        perplexities = []     # per-scale codebook perplexity (None for plain VQ)

        for k, scale in enumerate(self.scale_factors):
            r_k = residual
            if scale > 1:
                r_k = F.interpolate(r_k, size=(H // scale, W // scale),
                                    mode='bilinear', align_corners=False)

            code_k, emb_loss_k, info_k = self.quantizers[k](self.quant_convs[k](r_k))
            emb_losses.append(emb_loss_k)
            indices.append(info_k[2])
            perplexities.append(info_k[0])   # info[0] is None unless VectorQuantizerVQR

            if scale > 1:
                code_k = F.interpolate(code_k, size=(H, W),
                                       mode='bilinear', align_corners=False)
            latent_k = self.post_quant_convs[k](code_k)

            scale_latents.append(latent_k)
            residual = residual - latent_k

        # Normalize emb_loss to keep qloss magnitude stable regardless of num_scales
        self.vq_perplexity = perplexities   # stashed for backward_g logging
        return scale_latents, sum(emb_losses) / self.num_scales, indices

    def decode(self, latent):
        """(N, z_channels, H, W) combined latent -> (N, out_ch, Y, X) reconstruction.

        The latent is already in z_channels space (post_quant_convs applied
        per-scale in encode), so it feeds the decoder directly — no
        post_quant_conv needed here.
        """
        return self.decoder(latent)

    def forward(self, x):
        """(N, C, Y, X) slice batch -> (reconstruction, emb_loss, scale_latents)."""
        scale_latents, emb_loss, _ = self.encode(x)
        return self.decode(sum(scale_latents)), emb_loss, scale_latents

    def adv_loss_six_way(self, x, net_d, truth):
        """Score all three orthogonal slice stacks of a (B, C, Y, X, Z) volume with the
        same 2D discriminator; B is folded into each plane's slice batch."""
        B, C, Y, X, Z = x.shape
        loss = 0
        loss += self.add_loss_adv(a=x.permute(0, 2, 1, 4, 3).reshape(B * Y, C, Z, X),
                                  net_d=net_d, truth=truth)
        loss += self.add_loss_adv(a=x.permute(0, 3, 1, 4, 2).reshape(B * X, C, Z, Y),
                                  net_d=net_d, truth=truth)
        loss += self.add_loss_adv(a=x.permute(0, 4, 1, 2, 3).reshape(B * Z, C, Y, X),
                                  net_d=net_d, truth=truth)
        loss = loss / 3
        return loss

    def get_xy_plane(self, x):
        return self.vol_to_slices(x)

    def _z_transform(self, v):
        """Optional latent-Z reshaping (--downbranch / --resizebranch) on a
        (B, C, H, W, Z) volume before it enters net_g."""
        if self.hparams.downbranch > 1:
            v = nn.MaxPool3d((1, 1, self.hparams.downbranch))(v)
        if self.hparams.resizebranch != 1:
            v = nn.Upsample(scale_factor=(1, 1, self.hparams.resizebranch),
                            mode='trilinear')(v)
        return v

    def _netg_decode(self, scale_latents, batch_size, stop_at=None):
        """Shared net_g decode path for generation() / generation_test().

        In:  scale_latents — encode()'s per-scale list, (B*Z, z_channels, H, W).
        Out: net_g's output dict — 'out0' the full-res isotropic volume,
             'out128' / 'out64' the 1/2- and 1/4-res heads, each (B, 1, Y', X', Z').
        stop_at ('out64'|'out128', inference only): early-exit inside net_g right
        after that head — the dict then holds only the heads computed so far.
        Only passed through when set, so netGs without the kwarg keep working.

        Trunk: full sum (every code gets full decoder depth, like vqcleanM0aMS)
        → Z-transforms → per-slice decoder.conv_in → net_g. The two finest
        scales are ADDITIONALLY injected (inject_convs, at latent resolution)
        after the out64/out128 taps; net_g resizes them spatially at the
        injection points. They get the same Z-transforms so the grids match.
        """
        def prep(s):  # slice batch -> Z-transformed (B, z_channels, H, W, Z') volume
            return self._z_transform(self.slices_to_vol(s, batch_size))

        trunk = prep(sum(scale_latents))
        trunk = self.slices_to_vol(self.decoder.conv_in(self.vol_to_slices(trunk)),
                                   batch_size)
        inject = {'pre_up2': self.inject_convs[0](prep(scale_latents[-2])),
                  'pre_up1': self.inject_convs[1](prep(scale_latents[-1]))}
        if stop_at is None:
            return self.net_g(trunk, method='decode', inject=inject)
        return self.net_g(trunk, method='decode', inject=inject, stop_at=stop_at)

    def generation(self, batch, deterministic=False):
        if self.hparams.cropz > 0 and self.training:
            if deterministic:
                z_init = 0
            else:
                z_init = np.random.randint(batch['img'][0].shape[4] - self.hparams.cropz)
            for b in range(len(batch['img'])):
                batch['img'][b] = batch['img'][b][:, :, :, :, z_init:z_init + self.hparams.cropz]

        # Keep all non-input views (data1, data2, ...) at full Z — captured before the
        # `dsp` subsample so xcube/ycube retain their real HR-Z. Used for GIF logging
        # (and available for projection supervision). The `dsp`/`usp` reassignments
        # below replace batch['img'][b] with new tensors, so these references stay full-Z.
        self.aux_views = [batch['img'][b] for b in range(1, len(batch['img']))]

        if self.hparams.dsp > 1:
            if deterministic:
                z_init = 0
            else:
                z_init = np.random.randint(self.hparams.dsp)
            for b in range(len(batch['img'])):
                batch['img'][b] = batch['img'][b][:, :, :, :, z_init::self.hparams.dsp]

        if self.hparams.usp != 1:
            for b in range(len(batch['img'])):
                batch['img'][b] = nn.Upsample(scale_factor=(1, 1, self.hparams.usp),
                                              mode='trilinear')(batch['img'][b])

        self.oriX = batch['img'][0]

        # 2D VQ stack on the XY-slice batch
        slices = self.vol_to_slices(self.oriX)  # (B*Z, C, Y, X)
        if self.training:
            slices = slices.requires_grad_(True)

        scale_latents, self.qloss, _ = self.encode(slices)
        self.reconstructions = self.decode(sum(scale_latents))
        self.scale_latents = scale_latents   # per-scale contributions, kept for inspection/tests

        # 3D generator: shared trunk + lateral-inject path
        out = self._netg_decode(scale_latents, batch_size=self.oriX.shape[0])
        self.XupX = out['out0']         # full-res isotropic output
        self.XupX128 = out['out128']    # 1/2-res output head
        self.XupX64 = out['out64']      # 1/4-res output head
        # Extra GIF panels for MLflow logging (base.py upsamples these to the full grid
        # with trilinear interpolation so the panels concatenate cleanly).
        self.gif_scales = [self.XupX128, self.XupX64]
        # self.Xup = self.upsample(self.oriX)
        # Match Xup to the network output's actual size (not a fixed cropsize cube),
        # so validation without cropz (full z-depth) keeps Xup and XupX the same shape
        self.Xup = F.interpolate(self.oriX, size=self.XupX.shape[2:], mode='trilinear')

    def generation_test(self, x, method='full', stop_at=None):
        """Stateless inference API — one entry point, staged by `method`.

        method='full' (default): x = (B, C, Y, X, Z) normalized volume
            -> the full-res isotropic volume (what generation() assigns to
            self.XupX, i.e. net_g's 'out0'). Equals 'encode' then 'decode'.
        method='encode': x = (B, C, Y, X, Z) volume -> (scale_latents, indices),
            the compact VQ representation:
            scale_latents — list of num_scales per-scale contributions in volume
                form (B, z_channels, H, W, Z), all at full latent resolution;
            indices — list of (B, Z, hk, wk) int64 codebook indices,
                hk = H // scale_factors[k]: the Zarr-storable form; rebuild the
                latents exactly with latents_from_indices().
        method='decode': x = scale_latents list -> the 3D isotropic volume
            (B, 1, Y', X', Z') via the net_g trunk + lateral injects.
        method='decode_ms': x = scale_latents list -> net_g's full output dict
            ('out0' full res, 'out128' 1/2, 'out64' 1/4 — one shared decoder
            pass). stop_at='out64'|'out128' early-exits net_g after that head
            (cheaper; the dict then holds only the heads computed so far).
        method='reconstruction': x = scale_latents list -> the slice-wise 2D
            VQ-head reconstruction as a (B, out_ch, Y, X, Z) volume — no net_g,
            no Z upsampling (what training keeps as self.reconstructions).

        Contract (all methods):
        - Inputs are already normalized and at the model's expected Z
          resolution; none of generation()'s dsp/usp/cropz data-prep applies.
        - Stateless: nothing is stashed on self.
        - True batch support: Z is folded with B into the 2D batches; net_g's
          decode path is batch-agnostic.
        - Device/dtype/precision are the caller's job (works under .half() +
          autocast); wrap calls in torch.no_grad()/inference_mode().
        """
        assert not self.training, \
            'generation_test is eval-only (train mode would update BatchNorm running stats)'

        if method in ('full', 'encode'):
            B = x.shape[0]
            slice_latents, _, raw_indices = self.encode(self.vol_to_slices(x))
            if method == 'full':
                return self._netg_decode(slice_latents, batch_size=B)['out0']
            H, W = slice_latents[0].shape[-2:]
            Z = slice_latents[0].shape[0] // B
            scale_latents = [self.slices_to_vol(s, B) for s in slice_latents]
            # reshape works for every quantizer index layout (flat, (N,1) with
            # remap, or (N, hk, wk) with --sane_index_shape)
            indices = [idx.reshape(B, Z, H // scale, W // scale)
                       for scale, idx in zip(self.scale_factors, raw_indices)]
            return scale_latents, indices

        if method in ('decode', 'decode_ms'):
            B = x[0].shape[0]
            slice_latents = [self.vol_to_slices(v) for v in x]
            if method == 'decode':
                return self._netg_decode(slice_latents, batch_size=B)['out0']
            return self._netg_decode(slice_latents, batch_size=B, stop_at=stop_at)

        if method == 'reconstruction':
            B = x[0].shape[0]
            recon = self.decode(sum(self.vol_to_slices(v) for v in x))
            return self.slices_to_vol(recon, B)

        raise ValueError(f"generation_test: unknown method '{method}'")

    def latents_from_indices(self, indices):
        """Rebuild generation_test(method='encode')'s scale_latents exactly from
        stored codebook indices — the decode side of the compression path (no
        encoder run needed).

        In:  indices — list of num_scales (B, Z, hk, wk) int64 tensors.
        Out: scale_latents — list of (B, z_channels, H, W, Z) volume-form
             latents. The interpolate + post_quant_conv here match encode()
             steps 4-5 exactly; the only difference from the encode path's
             latents is float round-off (~1e-7) from its straight-through
             estimator (encode carries z + (z_q - z).detach(), this path the
             exact codebook values z_q).
        """
        assert not self.training, 'latents_from_indices is eval-only'
        H, W = indices[-1].shape[-2:]  # finest scale (scale_factor 1) = full latent res
        scale_latents = []
        for k, (scale, idx) in enumerate(zip(self.scale_factors, indices)):
            B, Z, hk, wk = idx.shape
            code_k = self.quantizers[k].get_codebook_entry(
                idx.reshape(-1), shape=(B * Z, hk, wk, self.embed_dim))
            if scale > 1:
                code_k = F.interpolate(code_k, size=(H, W),
                                       mode='bilinear', align_corners=False)
            latent_k = self.post_quant_convs[k](code_k)
            scale_latents.append(self.slices_to_vol(latent_k, B))
        return scale_latents

    def get_projection(self, x, depth, how='mean', uprate=None):
        # `uprate` only matters for how='dsp' (defaults to the full-res rate); the coarse
        # heads pass their own halved/quartered rate.
        if how == 'dsp':
            u = self.uprate if uprate is None else uprate
            x = x[:, :, :, :, (u // 2)::u * self.hparams.skipl1]
        else:
            x = x.unfold(-1, depth, depth)
            if how == 'mean':
                x = x.mean(dim=-1)
            elif how == 'max':
                x, _ = x.max(dim=-1)
        return x

    def backward_g(self):
        loss_g = 0
        loss_dict = {}

        axx = self.adv_loss_six_way(self.XupX, net_d=self.net_d, truth=True)

        loss_l1 = self.add_loss_l1(
            a=self.get_projection(self.XupX, depth=self.uprate * self.hparams.skipl1,
                                  how=self.hparams.l1how),
            b=self.oriX[:, :, :, :, ::self.hparams.skipl1]
        )

        if self.hparams.lbm_ms_ssim > 0:
            B, C, Y, X, Z = self.XupX.shape
            loss_ms_ssim = 1 - ms_ssim(
                self.XupX.permute(0, 2, 1, 4, 3).reshape(B * Y, C, Z, X),
                self.Xup.permute(0, 2, 1, 4, 3).reshape(B * Y, C, Z, X),
                data_range=2.0, size_average=True, win_size=7,
                weights=[0.0448, 0.2856, 0.6696]
            )
            loss_dict['ms_ssim'] = loss_ms_ssim
            loss_g += loss_ms_ssim * self.hparams.lbm_ms_ssim

        loss_dict['axx'] = axx
        loss_g += axx * self.hparams.adv
        loss_dict['l1'] = loss_l1
        loss_g += loss_l1 * self.hparams.lamb

        # MSskipE (F2): real-data supervision of the coarse heads — Z-project out128/out64
        # (same skipl1/--l1how as the main l1, with their halved/quartered uprate) and match
        # the XY-avg-pooled input.
        if self.hparams.lamb_coarse > 0:
            ori128 = F.avg_pool3d(self.oriX, (2, 2, 1))
            ori64 = F.avg_pool3d(self.oriX, (4, 4, 1))
            loss_l1_128 = self.add_loss_l1(
                a=self.get_projection(self.XupX128,
                                      depth=(self.uprate // 2) * self.hparams.skipl1,
                                      how=self.hparams.l1how, uprate=self.uprate // 2),
                b=ori128[:, :, :, :, ::self.hparams.skipl1])
            loss_l1_64 = self.add_loss_l1(
                a=self.get_projection(self.XupX64,
                                      depth=(self.uprate // 4) * self.hparams.skipl1,
                                      how=self.hparams.l1how, uprate=self.uprate // 4),
                b=ori64[:, :, :, :, ::self.hparams.skipl1])
            loss_dict['l1c'] = loss_l1_128 + loss_l1_64
            loss_g += (loss_l1_128 + loss_l1_64) * self.hparams.lamb * self.hparams.lamb_coarse

        # --- Multi-scale (progressive) output supervision ---
        # Pyramid consistency: coarse heads match avg-pooled full-res output.
        ref128 = F.avg_pool3d(self.XupX, 2)
        ref64 = F.avg_pool3d(self.XupX, 4)
        if self.hparams.pyr_detach:
            ref128 = ref128.detach()
            ref64 = ref64.detach()
        loss_pyr = self.add_loss_l1(self.XupX128, ref128) + self.add_loss_l1(self.XupX64, ref64)
        loss_dict['pyr'] = loss_pyr
        loss_g += loss_pyr * self.hparams.lamb_pyr

        # Per-scale six-way adversarial loss from the dedicated discriminators.
        # --adv_ms scales it on top of --adv; adv_ms=0 skips the coarse discriminators entirely.
        if self.hparams.adv_ms > 0:
            axx128 = self.adv_loss_six_way(self.XupX128, net_d=self.net_d_128, truth=True)
            axx64 = self.adv_loss_six_way(self.XupX64, net_d=self.net_d_64, truth=True)
            loss_dict['axx128'] = axx128
            loss_dict['axx64'] = axx64
            loss_g += (axx128 + axx64) * self.hparams.adv * self.hparams.adv_ms

        oriX_slices = self.vol_to_slices(self.oriX)  # (B*Z, C, Y, X), matches reconstructions

        aeloss, log_dict_ae = self.loss(
            self.qloss, oriX_slices, self.reconstructions,
            0, self.global_step,
            last_layer=self.get_last_layer(), split="train"
        )
        loss_g += aeloss
        loss_dict['ae'] = aeloss
        loss_dict['vq'] = self.qloss
        loss_g += self.qloss

        # per-scale codebook perplexity (effective # codes in use) — logged only
        # when the quantizer reports it (VectorQuantizerVQR); metric only, not a loss
        for k, ppl in enumerate(getattr(self, 'vq_perplexity', []) or []):
            if ppl is not None:
                loss_dict['ppl%d' % k] = ppl.detach()

        loss_dict['sum'] = loss_g
        return loss_dict

    def backward_d(self):
        loss_d = 0
        loss_dict = {}

        dxx = self.adv_loss_six_way(self.XupX, net_d=self.net_d, truth=False)
        dx = self.add_loss_adv(a=self.get_xy_plane(self.oriX), net_d=self.net_d, truth=True)

        loss_dict['dxx_x'] = dxx + dx
        loss_d += (dxx + dx) * self.hparams.adv

        # Per-scale discriminator updates: fakes from the coarse heads, real targets are the
        # real X/Y slices downsampled to the matching resolution. Skipped when --adv_ms=0.
        if self.hparams.adv_ms > 0:
            real_xy = self.get_xy_plane(self.oriX)
            d128 = self.adv_loss_six_way(self.XupX128, net_d=self.net_d_128, truth=False) \
                + self.add_loss_adv(a=F.avg_pool2d(real_xy, 2), net_d=self.net_d_128, truth=True)
            d64 = self.adv_loss_six_way(self.XupX64, net_d=self.net_d_64, truth=False) \
                + self.add_loss_adv(a=F.avg_pool2d(real_xy, 4), net_d=self.net_d_64, truth=True)
            loss_dict['d128'] = d128
            loss_dict['d64'] = d64
            loss_d += (d128 + d64) * self.hparams.adv * self.hparams.adv_ms

        oriX_slices = self.vol_to_slices(self.oriX)  # (B*Z, C, Y, X), matches reconstructions

        discloss, log_dict_disc = self.loss(
            self.qloss, oriX_slices, self.reconstructions,
            1, self.global_step,
            last_layer=self.get_last_layer(), split="train"
        )
        loss_d += discloss
        loss_dict['disc'] = discloss
        loss_dict['sum'] = loss_d
        return loss_dict

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self)

    # MSskipE (F1): run the whole validation loop (metrics, KID, GIFs) under EMA weights.
    def on_validation_start(self):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)

    def on_validation_end(self):
        if self.use_ema:
            self.model_ema.restore(self.parameters())

    def training_epoch_end(self, outputs):
        # The epoch checkpoints (saved by BaseModel.training_epoch_end via netg_names) should
        # hold EMA weights too — swap them in around the base implementation.
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            try:
                super().training_epoch_end(outputs)
            finally:
                self.model_ema.restore(self.parameters())
        else:
            super().training_epoch_end(outputs)

    def configure_optimizers(self):
        lr_d = self.hparams.lr
        lr_g = getattr(self.hparams, 'lr_g_factor', 1.0) * self.hparams.lr
        print("lr_d", lr_d)
        print("lr_g", lr_g)

        opt_ae = torch.optim.Adam(
            list(self.encoder.parameters()) +
            list(self.decoder.parameters()) +
            list(self.quantizers.parameters()) +
            list(self.quant_convs.parameters()) +
            list(self.post_quant_convs.parameters()) +
            list(self.inject_convs.parameters()) +
            list(self.net_g.parameters()),
            lr=lr_g, betas=(0.5, 0.9)
        )

        opt_disc = torch.optim.Adam(
            list(self.loss.discriminator.parameters()) +
            list(self.net_d.parameters()) +
            list(self.net_d_128.parameters()) +
            list(self.net_d_64.parameters()),
            lr=lr_d, betas=(0.5, 0.9)
        )

        if hasattr(self.hparams, 'scheduler_config') and self.hparams.scheduler_config is not None:
            scheduler = instantiate_from_config(self.hparams.scheduler_config)
            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {'scheduler': LambdaLR(opt_ae, lr_lambda=scheduler.schedule),
                 'interval': 'step', 'frequency': 1},
                {'scheduler': LambdaLR(opt_disc, lr_lambda=scheduler.schedule),
                 'interval': 'step', 'frequency': 1},
            ]
            self.net_g_scheduler = scheduler[0]['scheduler']
            self.net_d_scheduler = scheduler[1]['scheduler']
            return [opt_ae, opt_disc], scheduler

        self.net_g_scheduler = get_scheduler(opt_ae, self.hparams)
        self.net_d_scheduler = get_scheduler(opt_disc, self.hparams)
        return [opt_ae, opt_disc], []

    def save_checkpoint(self, filepath):
        state_dict = {}

        for k, v in self.encoder.state_dict().items():
            state_dict[f'encoder.{k}'] = v
        for k, v in self.decoder.state_dict().items():
            state_dict[f'decoder.{k}'] = v
        for k, v in self.quantizers.state_dict().items():
            state_dict[f'quantizers.{k}'] = v
        for k, v in self.quant_convs.state_dict().items():
            state_dict[f'quant_convs.{k}'] = v
        for k, v in self.post_quant_convs.state_dict().items():
            state_dict[f'post_quant_convs.{k}'] = v
        for k, v in self.inject_convs.state_dict().items():
            state_dict[f'inject_convs.{k}'] = v
        for k, v in self.discriminator.state_dict().items():
            state_dict[f'loss.discriminator.{k}'] = v

        if self.use_ema:
            for k, v in self.model_ema.state_dict().items():
                state_dict[f'model_ema.{k}'] = v

        checkpoint = {
            "state_dict": state_dict,
            "global_step": self.global_step,
            "optimizer_g": self.optimizer_g.state_dict(),
            "optimizer_d": self.optimizer_d.state_dict(),
        }

        if hasattr(self, 'hparams'):
            checkpoint['hparams'] = self.hparams

        torch.save(checkpoint, filepath)
        print(f"VQGAN model saved to {filepath}")
