from models.MScleanQDPSF import GAN as MScleanQDPSFGAN
import torch
import torch.nn as nn
import torch.nn.functional as F


# MScleanQDPSFX: MScleanQDPSF + the two stage-4 audit extensions, each off by
# default (both off = behaviorally identical to MScleanQDPSF).
#
# --inject_x  Decoder-stage input injection. The generator's only input is the
#   per-slice quantized latent; quantization destroys the small intensity
#   gradations (partial-volume / PSF ratios across neighboring slices) that
#   carry sub-Nyquist axial position, and fine lateral detail is
#   re-hallucinated from 16x16 codes (record/2026-11-26-stage4-next-models.md,
#   proposal 1). Fix: feed the measured volume itself into net_g's existing
#   lateral-inject points (pre_up2 at 1/4 res, pre_up1 at 1/2 res), through
#   small conv adapters whose output conv is zero-initialized
#   (ControlNet-style, same as inject_convs), so training starts numerically
#   identical to the parent and can only learn helpful evidence paths. The VQ
#   path stays untouched as the structural prior. Unlike --residual (which
#   anchored the OUTPUT and collapsed onto interpolation while the adversarial
#   gradient hallucinated texture, skip_r1), the evidence here arrives before
#   the up2/up1 conv stages and the final tanh, so every loss still shapes how
#   it is used.
#   Inference: generation_test(method='full') injects the input volume;
#   method='decode'/'decode_ms' (latents only, no measured volume available)
#   falls back to the parent path WITHOUT injection — do not evaluate an
#   --inject_x checkpoint through the latents-only path.
#
# --r1 GAMMA (with --r1_every N)  Lazy R1 gradient penalty on the real XY
#   slices for the full-resolution discriminator net_d (proposal 2):
#   (gamma/2) * E[|grad_x D(x_real)|^2], computed every N d-steps and scaled
#   by N. Targets D saturation/overfit (the diagnosed cause of late z_res
#   decay). Combine with TTUR via the existing --lr_g_factor (e.g. 0.5: G at
#   lr/2, D at lr).
#
# --d_hinge  Hinge adversarial loss for every add_loss_adv discriminator
#   (net_d, and net_d_mp when present) instead of vanilla BCE: D-real
#   relu(1-logits), D-fake relu(1+logits), G -logits (non-saturating). The VQ
#   stack's internal VQLPIPSWithDiscriminator already runs hinge and is
#   untouched. backward_g/backward_d mark which phase is running so the
#   truth=True ambiguity (G-fake vs D-real) resolves correctly.
#
# --mp_d N (with --adv_mp W)  Extra multi-plane (2.5D) patch discriminator
#   judging N adjacent planes jointly as an N-channel image. A single-plane D
#   cannot see inter-plane consistency (axial flicker, wrong partial-volume
#   progression) — exactly the signature of fabricated axial detail. Fakes:
#   stacks of N adjacent planes of the generated volume along each of Y/X/Z
#   (1 px spacing). Reals: N copies of a real XY slice shifted 1 px along Y
#   (and along X), whose joint statistics match true 1-px-spaced planes by
#   in-plane isotropy (SliceGAN-style). Weight --adv * --adv_mp. When --r1 is
#   on, the same lazy R1 penalty is applied to net_d_mp on its real stacks.
#   net_d_mp is saved via netd_names and EMA-registered like every other D
#   (LitEma.forward crashes on any requires_grad param missing from its map).
#
# --band_split SIGMA  Frequency-separated adversarial supervision (DSGAN-
#   style): every input to the axial-SR discriminators (six-way fakes, real
#   XY slices, multi-plane stacks, R1 reals) is high-passed a - G_sigma(a)
#   (separable 2D Gaussian, reflect pad) before the D sees it. The PSF replay
#   term already owns the low band (it blurs with the PSF), so this stops the
#   adversarial gradient from fighting over the band the data term supervises
#   and spends all its capacity in the beyond-Nyquist band. The generator
#   OUTPUT is untouched (unlike the failed --residual: nothing is anchored to
#   interpolation — the split is a loss-side filter on the D inputs only).
#   The VQ stack's internal reconstruction discriminator is untouched.
#
# --g_noise  StyleGAN-style decoder noise: i.i.d. Gaussian noise with a
#   learnable scalar scale (init 0) added to the net_g features at the
#   pre_up2 / pre_up1 inject points, TRAINING ONLY (eval is deterministic:
#   codes remain the sole information source — the noise carries no sample
#   information, so the compression claim is intact). Gives the generator a
#   sampling mechanism for the stochastic texture component the D demands,
#   instead of averaging it out.
#
# --refine_hr N  Decode-tail capacity: N residual 3x3x3 conv blocks (GN +
#   LeakyReLU, final conv zero-init) at FULL resolution between net_g's up1
#   and the conv7 output heads. Today the entire beyond-Nyquist band is
#   synthesized by the single conv7_k conv after the last upsample; this adds
#   receptive field and depth where the new axial detail must be placed.
#   Implemented by calling net_g's own submodules in the exact forward order
#   (up3/conv5/out64/inject/up2/conv6/out128/inject/up1), so with the final
#   convs zero-init the output starts bit-identical to the parent. Inputs are
#   code-derived features only — claim-intact.
#
# --zshuffle  Learned axial sub-pixel head: a conv on the post-conv6 half-res
#   features expands channels x2, pixel-shuffles the factor into Z (each
#   output z-phase gets its own filter instead of shared transposed-conv
#   taps), trilinear-resizes laterally and enters the full-res tail through a
#   zero-init 1x1x1 adapter. Counters the low-pass bias of the transposed-
#   conv upsampler for the finest axial band. Code-derived only — claim-
#   intact.
#
# NOTE (snapshot): train.py copies only this file into the checkpoint dir; it
# subclasses models/MScleanQDPSF.py -> MScleanQD.py -> MSclean.py.

class GAN(MScleanQDPSFGAN):
    def __init__(self, hparams, train_loader, eval_loader, checkpoints):
        MScleanQDPSFGAN.__init__(self, hparams, train_loader, eval_loader, checkpoints)
        self._inject_src = None
        if getattr(self.hparams, 'inject_x', False):
            nf = self.hparams.ngf

            def adapter(c_out):
                out = nn.Conv3d(nf, c_out, 3, padding=1)
                nn.init.zeros_(out.weight)
                nn.init.zeros_(out.bias)
                return nn.Sequential(nn.Conv3d(1, nf, 3, padding=1),
                                     nn.LeakyReLU(0.2, inplace=True), out)

            # Measured-volume adapters for net_g's inject points: [0] 'pre_up2'
            # (4*nf channels, 1/4 res), [1] 'pre_up1' (2*nf channels, 1/2 res).
            self.xinj_convs = nn.ModuleList([adapter(4 * nf), adapter(2 * nf)])
            self.netg_names['xinj_convs'] = 'xinj_convs'
            # Re-register EMA so the new generator-side params are tracked like
            # the rest (LitEma snapshots named params at construction).
            from ldm.modules.ema import LitEma
            self.model_ema = LitEma(self, decay=getattr(hparams, 'ema_decay', 0.9999))
            # Rebuild so opt_ae includes xinj_convs from the start (same
            # pattern as MScleanQD's net_d_prefix).
            self.configure_optimizers()
        self._adv_phase = 'g'
        if getattr(self.hparams, 'mp_d', 0) > 0:
            # Multi-plane D: same patch D as net_d with input_nc = mp_d planes.
            from networks.registry import network_registry
            self.net_d_mp = network_registry.get_discriminator(
                self.hparams.netD, input_nc=self.hparams.mp_d,
                output_nc=self.hparams.output_nc, norm=self.hparams.norm,
                norm_layer=nn.BatchNorm2d, ndf=self.hparams.ndf)
            self.netd_names['net_d_mp'] = 'net_d_mp'
            # LitEma.forward requires EVERY requires_grad param in its name map
            # (all existing Ds are tracked too — they predate LitEma in
            # MSclean.__init__), so re-register after adding net_d_mp.
            from ldm.modules.ema import LitEma
            self.model_ema = LitEma(self, decay=getattr(hparams, 'ema_decay', 0.9999))
            self.configure_optimizers()
        if getattr(self.hparams, 'band_split', 0) > 0:
            sig = float(self.hparams.band_split)
            r = max(1, int(3.0 * sig + 0.5))
            t = torch.arange(-r, r + 1, dtype=torch.float32)
            k = torch.exp(-0.5 * (t / sig) ** 2)
            self.register_buffer('band_kernel', k / k.sum())
        extra_g = False
        nf = self.hparams.ngf
        if getattr(self.hparams, 'g_noise', False):
            self.gnoise_scales = nn.ParameterList(
                [nn.Parameter(torch.zeros(1)), nn.Parameter(torch.zeros(1))])
            self.netg_names['gnoise_scales'] = 'gnoise_scales'
            extra_g = True
        if getattr(self.hparams, 'zshuffle', False):
            self.zshuf_conv = nn.Conv3d(2 * nf, 2 * nf, 3, padding=1)
            self.zshuf_adapt = nn.Conv3d(nf, nf, 1)
            nn.init.zeros_(self.zshuf_adapt.weight)
            nn.init.zeros_(self.zshuf_adapt.bias)
            self.netg_names['zshuf_conv'] = 'zshuf_conv'
            self.netg_names['zshuf_adapt'] = 'zshuf_adapt'
            extra_g = True
        if getattr(self.hparams, 'refine_hr', 0) > 0:
            def rblock():
                last = nn.Conv3d(nf, nf, 3, padding=1)
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)
                return nn.Sequential(nn.Conv3d(nf, nf, 3, padding=1),
                                     nn.GroupNorm(8, nf),
                                     nn.LeakyReLU(0.2, inplace=True), last)
            self.refine_blocks = nn.ModuleList(
                [rblock() for _ in range(self.hparams.refine_hr)])
            self.netg_names['refine_blocks'] = 'refine_blocks'
            extra_g = True
        if extra_g:
            # LitEma.forward requires every requires_grad param in its map.
            from ldm.modules.ema import LitEma
            self.model_ema = LitEma(self, decay=getattr(hparams, 'ema_decay', 0.9999))
            self.configure_optimizers()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parent_parser = MScleanQDPSFGAN.add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("MScleanQDPSFX")
        parser.add_argument("--inject_x", action='store_true', default=False,
                            help='feed the measured volume into the decoder up-stages '
                                 '(zero-init adapters at the pre_up2/pre_up1 inject points)')
        parser.add_argument("--r1", type=float, default=0.0,
                            help='R1 gradient-penalty weight gamma on real XY slices for net_d; 0 disables')
        parser.add_argument("--r1_every", type=int, default=16,
                            help='apply the (lazy) R1 penalty every this many d-steps, scaled by the interval')
        parser.add_argument("--d_hinge", action='store_true', default=False,
                            help='hinge adversarial loss (D: relu margins, G: -logits) for all '
                                 'add_loss_adv discriminators instead of vanilla BCE')
        parser.add_argument("--mp_d", type=int, default=0,
                            help='extra multi-plane patch D judging this many adjacent planes '
                                 'jointly (N-channel input); 0 disables')
        parser.add_argument("--adv_mp", type=float, default=1.0,
                            help='weight of the multi-plane adversarial loss (times --adv)')
        parser.add_argument("--band_split", type=float, default=0.0,
                            help='sigma (px) of the 2D Gaussian low-pass; all axial-SR '
                                 'discriminator inputs are high-passed a - G_sigma(a); 0 disables')
        parser.add_argument("--g_noise", action='store_true', default=False,
                            help='learnable-scale Gaussian noise at the decoder inject points, '
                                 'training only (eval deterministic)')
        parser.add_argument("--refine_hr", type=int, default=0,
                            help='number of full-res residual 3x3x3 blocks between net_g up1 '
                                 'and the output heads (zero-init); 0 disables')
        parser.add_argument("--zshuffle", action='store_true', default=False,
                            help='learned axial sub-pixel (Z pixel-shuffle) branch from the '
                                 'half-res features into the full-res tail (zero-init adapter)')
        return parent_parser

    def configure_optimizers(self):
        opts, sched = super().configure_optimizers()
        from networks.networks import get_scheduler
        # lr/betas come from opt_ae's defaults (lr_g, (0.5, 0.9)); rebuild the
        # scheduler after adding groups so its base_lrs cover them.
        g_added = False
        for name in ('xinj_convs', 'gnoise_scales', 'zshuf_conv', 'zshuf_adapt',
                     'refine_blocks'):
            if hasattr(self, name):
                opts[0].add_param_group({'params': list(getattr(self, name).parameters())})
                g_added = True
        if g_added and not sched:
            self.net_g_scheduler = get_scheduler(opts[0], self.hparams)
        if hasattr(self, 'net_d_mp'):
            opts[1].add_param_group({'params': list(self.net_d_mp.parameters())})
            if not sched:
                self.net_d_scheduler = get_scheduler(opts[1], self.hparams)
        return opts, sched

    # ------------------------------------------------------------------ hinge

    def add_loss_adv(self, a, net_d, truth):
        if getattr(self.hparams, 'band_split', 0) > 0:
            a = self._band_hp(a)
        if not getattr(self.hparams, 'd_hinge', False):
            return super().add_loss_adv(a, net_d, truth)
        logits = net_d(a)[0]
        if self._adv_phase == 'd':
            return F.relu(1.0 - logits).mean() if truth else F.relu(1.0 + logits).mean()
        return -logits.mean()  # generator phase (non-saturating)

    def _band_hp(self, a):
        """High-pass for D inputs: a - G_sigma(a), separable 2D Gaussian over
        the last two dims, reflect pad, per channel. (N, C, H, W)."""
        k = self.band_kernel.to(dtype=a.dtype)
        C = a.shape[1]
        r = (k.numel() - 1) // 2
        w_h = k.view(1, 1, -1, 1).expand(C, 1, -1, 1)
        w_w = k.view(1, 1, 1, -1).expand(C, 1, 1, -1)
        low = F.conv2d(F.pad(a, (0, 0, r, r), mode='reflect'), w_h, groups=C)
        low = F.conv2d(F.pad(low, (r, r, 0, 0), mode='reflect'), w_w, groups=C)
        return a - low

    # ------------------------------------------------------- multi-plane D

    def _mp_fake(self, x):
        """Stacks of mp_d adjacent planes of a (B, 1, Y, X, Z) volume along
        each of Y/X/Z (1 px spacing), as N-channel images."""
        n = self.hparams.mp_d
        B, C, Y, X, Z = x.shape
        outs = []
        for v in (x[:, 0].permute(0, 1, 3, 2),    # (B, Y, Z, X): stack along Y
                  x[:, 0].permute(0, 2, 3, 1),    # (B, X, Z, Y): stack along X
                  x[:, 0].permute(0, 3, 1, 2)):   # (B, Z, Y, X): stack along Z
            # v: (B, A, H, W) -> (B, A-n+1, n, H, W) -> (B*(A-n+1), n, H, W)
            u = v.unfold(1, n, 1).permute(0, 1, 4, 2, 3)
            outs.append(u.reshape(-1, n, u.shape[3], u.shape[4]))
        return outs

    def _mp_real(self, x):
        """Real multi-plane stacks from XY slices: n copies shifted 1 px along
        Y (and along X). By in-plane isotropy their joint statistics match true
        1-px-spaced adjacent planes (SliceGAN-style)."""
        n = self.hparams.mp_d
        s = self.get_xy_plane(x)  # (B*Z, 1, Y, X)
        BZ, C, Y, X = s.shape
        return [torch.cat([s[:, :, k:Y - n + 1 + k, :] for k in range(n)], 1),
                torch.cat([s[:, :, :, k:X - n + 1 + k] for k in range(n)], 1)]

    def backward_g(self):
        self._adv_phase = 'g'
        loss_dict = super().backward_g()
        if getattr(self.hparams, 'mp_d', 0) > 0:
            axmp = sum(self.add_loss_adv(a, self.net_d_mp, truth=True)
                       for a in self._mp_fake(self.XupX)) / 3
            loss_dict['axmp'] = axmp
            loss_dict['sum'] = loss_dict['sum'] + axmp * self.hparams.adv * self.hparams.adv_mp
        return loss_dict

    def _netg_decode(self, scale_latents, batch_size, stop_at=None):
        hp = self.hparams
        use_hr = getattr(hp, 'refine_hr', 0) > 0 or getattr(hp, 'zshuffle', False)
        use_noise = getattr(hp, 'g_noise', False) and self.training
        # training: generation() has just set self.oriX; eval: generation_test
        # stashes its input in _inject_src (latents-only decode has neither ->
        # no injection; eval never uses a stale training oriX).
        src = None
        if getattr(hp, 'inject_x', False):
            src = self._inject_src if self._inject_src is not None else getattr(self, 'oriX', None)
            if not self.training and self._inject_src is None:
                src = None
        if not (use_hr or use_noise or src is not None):
            return super()._netg_decode(scale_latents, batch_size, stop_at)

        def prep(s):
            return self._z_transform(self.slices_to_vol(s, batch_size))

        trunk = prep(sum(scale_latents))
        trunk = self.slices_to_vol(self.decoder.conv_in(self.vol_to_slices(trunk)),
                                   batch_size)
        inj2 = self.inject_convs[0](prep(scale_latents[-2]))
        inj1 = self.inject_convs[1](prep(scale_latents[-1]))
        if src is not None:
            # Resize the measured volume to the decode-stage grids (net_g
            # up-stages double Y/X/Z each: pre_up2 = (Y/4, X/4, 2Z), pre_up1 =
            # (Y/2, X/2, 4Z)) and sum the adapter features with the latent
            # injects on that grid; net_g's own interpolate at the inject
            # point is then (near-)identity.
            B, C, Y, X, Z = src.shape
            x4 = F.interpolate(src, size=(Y // 4, X // 4, 2 * Z), mode='trilinear', align_corners=False)
            x2 = F.interpolate(src, size=(Y // 2, X // 2, 4 * Z), mode='trilinear', align_corners=False)
            inj2 = self.xinj_convs[0](x4) + F.interpolate(
                inj2, size=x4.shape[2:], mode='trilinear', align_corners=False)
            inj1 = self.xinj_convs[1](x2) + F.interpolate(
                inj1, size=x2.shape[2:], mode='trilinear', align_corners=False)
        if use_noise:
            # i.i.d. noise at the exact stage grids (interpolating noise would
            # low-pass it); scales init 0 -> starts as a no-op.
            nf = hp.ngf
            tB, _, tY, tX, tZ = trunk.shape
            g2 = (2 * tY, 2 * tX, 2 * tZ)
            g1 = (4 * tY, 4 * tX, 4 * tZ)
            inj2 = F.interpolate(inj2, size=g2, mode='trilinear', align_corners=False) \
                + self.gnoise_scales[0] * torch.randn(tB, 4 * nf, *g2,
                                                      device=trunk.device, dtype=trunk.dtype)
            inj1 = F.interpolate(inj1, size=g1, mode='trilinear', align_corners=False) \
                + self.gnoise_scales[1] * torch.randn(tB, 2 * nf, *g1,
                                                      device=trunk.device, dtype=trunk.dtype)
        inject = {'pre_up2': inj2, 'pre_up1': inj1}
        if not use_hr or stop_at is not None:
            # stop_at early-exits return only coarse heads, which the HR tail
            # does not touch -> net_g's own path is exact.
            if stop_at is None:
                return self.net_g(trunk, method='decode', inject=inject)
            return self.net_g(trunk, method='decode', inject=inject, stop_at=stop_at)
        return self._decode_hr_tail(trunk, inject)

    def _decode_hr_tail(self, trunk, inject):
        """Full decode mirroring Generator.forward's decode stage order
        exactly (ed023eMSfpn), with the full-res tail extended by the
        zshuffle branch and/or refine blocks (all zero-init: bit-identical to
        the parent at initialization)."""
        g = self.net_g
        x = g.up3(trunk)
        x = g.conv5(x)
        out64 = g.conv7_64(x)
        x = x + F.interpolate(inject['pre_up2'], size=x.shape[2:],
                              mode='trilinear', align_corners=False)
        x = g.up2(x)
        x = g.conv6(x)
        out128 = g.conv7_128(x)
        h128 = x
        x = x + F.interpolate(inject['pre_up1'], size=x.shape[2:],
                              mode='trilinear', align_corners=False)
        x = g.up1(x)
        if getattr(self.hparams, 'zshuffle', False):
            nf = self.hparams.ngf
            u = self.zshuf_conv(h128)  # (B, 2*nf, Y2, X2, Z2)
            B, C2, Y2, X2, Z2 = u.shape
            # move the channel factor 2 into Z: each output z-phase gets its
            # own filter (sub-pixel conv along Z)
            u = u.reshape(B, nf, 2, Y2, X2, Z2).permute(0, 1, 3, 4, 5, 2) \
                 .reshape(B, nf, Y2, X2, 2 * Z2)
            u = F.interpolate(u, size=x.shape[2:], mode='trilinear', align_corners=False)
            x = x + self.zshuf_adapt(u)
        for blk in getattr(self, 'refine_blocks', []):
            x = x + blk(x)
        return {'out0': g.conv7_k(x), 'out1': g.conv7_g(x),
                'out128': out128, 'out64': out64}

    def generation_test(self, x, method='full', stop_at=None):
        if getattr(self.hparams, 'inject_x', False) and method == 'full':
            self._inject_src = x
            try:
                return super().generation_test(x, method, stop_at)
            finally:
                self._inject_src = None
        return super().generation_test(x, method, stop_at)

    @staticmethod
    def _r1_term(net_d, real):
        logits = net_d(real)[0]
        grad = torch.autograd.grad(logits.sum(), real, create_graph=True)[0]
        return grad.pow(2).reshape(grad.shape[0], -1).sum(1).mean()

    def backward_d(self):
        self._adv_phase = 'd'
        loss_dict = super().backward_d()
        if getattr(self.hparams, 'mp_d', 0) > 0:
            dmp = sum(self.add_loss_adv(a, self.net_d_mp, truth=False)
                      for a in self._mp_fake(self.XupX)) / 3 \
                + sum(self.add_loss_adv(a, self.net_d_mp, truth=True)
                      for a in self._mp_real(self.oriX)) / 2
            loss_dict['dmp'] = dmp
            loss_dict['sum'] = loss_dict['sum'] + dmp * self.hparams.adv * self.hparams.adv_mp
        gamma = getattr(self.hparams, 'r1', 0.0)
        if gamma > 0 and self.global_step % self.hparams.r1_every == 0:
            def d_input(a):
                # R1 penalizes the gradient w.r.t. what the D actually sees:
                # under --band_split that is the high-passed input.
                a = a.detach()
                if getattr(self.hparams, 'band_split', 0) > 0:
                    a = self._band_hp(a)
                return a.requires_grad_(True)
            real = d_input(self.get_xy_plane(self.oriX))
            r1 = self._r1_term(self.net_d, real)
            if getattr(self.hparams, 'mp_d', 0) > 0:
                reals = [d_input(a) for a in self._mp_real(self.oriX)]
                r1 = r1 + sum(self._r1_term(self.net_d_mp, a) for a in reals) / 2
            loss_dict['r1'] = r1.detach()
            loss_dict['sum'] = loss_dict['sum'] + (gamma / 2) * self.hparams.r1_every * r1
        return loss_dict
