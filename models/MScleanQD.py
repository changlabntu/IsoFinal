from models.MSclean import GAN as MScleanGAN
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim.lr_scheduler import LambdaLR
from networks.networks import get_scheduler
from ldm.util import instantiate_from_config


# MScleanQD: MSclean + quantizer dropout (Encodec/SoundStream-style prefix-sum
# training) for the 2D decoder.
#
# Problem this fixes: MSclean's 2D decoder only ever sees the FULL sum of the
# residual-VQ scale latents, so decoding a truncated codec (prefix sum of the
# coarse scales — the natural variable-rate operating point) is out of
# distribution and produces junk/blur. MSclean's multi-scale machinery
# (out128/out64, net_d_128/net_d_64) is multi-scale OUT (coarse-resolution
# heads of a full-codec decode); this model adds multi-scale IN.
#
# Mechanism: with probability --p_prefix per generation() call, additionally
# decode sum(scale_latents[:k]) for a random k in {1..num_scales-1} and
# supervise it against a band-limited target — the input XY slices avg-pooled
# by the finest kept scale factor and upsampled back ("decode a CLEAN coarse
# image", the codec-side analog of --lamb_coarse). Losses:
#   l1p    — L1 to the band-limited target        (weight --lamb * --lamb_prefix)
#   lpipsp — LPIPS to the same target             (weight --lpips_prefix)
#   axp/dp — optional adversarial realism from a dedicated patch discriminator
#            net_d_prefix, reals = band-limited real slices (weight
#            --adv * --adv_prefix; 0 skips the discriminator entirely)
# Side effect (wanted): the coarse quantizers get a loss they alone must
# satisfy, pressuring codebook usage on the collapse-prone coarse scales.
#
# --p_prefix 0 (with --adv_prefix 0) makes this model behaviorally identical
# to MSclean. The prefix draw is per generation() call, so the g- and d-steps
# of a batch draw independently — fine, each step is self-consistent. On
# non-prefix steps net_d_prefix and the prefix losses are unused; the legacy
# pl.Trainer(strategy='ddp') default find_unused_parameters=True covers that.
#
# NOTE (snapshot): train.py copies only models/MScleanQD.py into the
# checkpoint dir; this file subclasses models/MSclean.py, so reproducing from
# a snapshot also needs MSclean.py at the run's logged git hash.
#
# This does NOT train net_g on truncated codecs — the 3D trunk still consumes
# the full sum. For truncated 3D decode experiments use prefix_latents()
# (zero-fills dropped scales so the lateral-inject positions stay aligned).
class GAN(MScleanGAN):
    def __init__(self, hparams, train_loader, eval_loader, checkpoints):
        MScleanGAN.__init__(self, hparams, train_loader, eval_loader, checkpoints)

        self.recon_prefix = None
        self.prefix_scale = None

        if getattr(self.hparams, 'adv_prefix', 0) > 0:
            # Dedicated discriminator for the prefix reconstructions, following
            # the net_d_128/net_d_64 pattern. Created after LitEma, so it is
            # not EMA-tracked — like all discriminators it is unused in
            # validation, and ema store/restore round-trips it untouched.
            self.net_d_prefix = self.set_networks(net='d')
            self.netd_names['net_d_prefix'] = 'net_d_prefix'
            # Rebuild so opt_disc includes net_d_prefix from the start (PL calls
            # configure_optimizers() again at fit; this keeps self.optimizer_d
            # and the schedulers consistent in the meantime).
            self.configure_optimizers()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parent_parser = MScleanGAN.add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("MScleanQD")
        parser.add_argument("--p_prefix", type=float, default=0.5,
                            help='probability per step of decoding a random coarse prefix '
                                 'of the codec scales; 0 recovers MSclean exactly')
        parser.add_argument("--lamb_prefix", type=float, default=1.0,
                            help='weight of the prefix L1 (times --lamb)')
        parser.add_argument("--lpips_prefix", type=float, default=1.0,
                            help='weight of the prefix LPIPS (via the vqperceptual LPIPS net); 0 disables')
        parser.add_argument("--adv_prefix", type=float, default=0.0,
                            help='weight of the prefix adversarial loss (times --adv) from a '
                                 'dedicated discriminator; 0 skips the discriminator entirely')
        parser.add_argument("--prefix_target", type=str, default='pool', choices=['pool', 'orig'],
                            help="prefix supervision target: 'pool' = input band-limited to the "
                                 "finest kept scale (clean blur), 'orig' = sharp input (Encodec-style, "
                                 "pushes the decoder to hallucinate detail)")
        return parent_parser

    def _prefix_target(self):
        """Supervision target for the prefix decode, as an XY-slice batch.

        'pool': the input slices avg-pooled by the finest kept scale factor and
        upsampled back — the frequency content a k-scale prefix can actually
        carry. 'orig': the sharp input slices.
        """
        x = self.vol_to_slices(self.oriX)
        s = self.prefix_scale
        if self.hparams.prefix_target == 'orig' or s == 1:
            return x
        return F.interpolate(F.avg_pool2d(x, s), size=x.shape[-2:],
                             mode='bilinear', align_corners=False)

    def prefix_latents(self, scale_latents, k):
        """Zero-fill all but the k coarsest scales, keeping list positions —
        for eval-time truncated-codec experiments. Positional consumers
        (_netg_decode's scale_latents[-2]/[-1] lateral injects) stay aligned,
        and sum() equals the true prefix sum."""
        return [s if i < k else torch.zeros_like(s)
                for i, s in enumerate(scale_latents)]

    def generation(self, batch, deterministic=False):
        super().generation(batch, deterministic)
        self.recon_prefix = None
        if self.training and getattr(self.hparams, 'p_prefix', 0) > 0 \
                and np.random.rand() < self.hparams.p_prefix:
            # keep the k coarsest scales; never the full sum (already supervised)
            k = np.random.randint(1, self.num_scales)
            self.prefix_scale = self.scale_factors[k - 1]
            self.recon_prefix = self.decode(sum(self.scale_latents[:k]))

    def backward_g(self):
        loss_dict = super().backward_g()
        if self.recon_prefix is not None:
            target = self._prefix_target()
            loss_prefix = 0

            l1p = self.add_loss_l1(self.recon_prefix, target)
            loss_dict['l1p'] = l1p
            loss_prefix += l1p * self.hparams.lamb * self.hparams.lamb_prefix

            if self.hparams.lpips_prefix > 0:
                lpipsp = self.loss.perceptual_loss(
                    target.contiguous(), self.recon_prefix.contiguous()).mean()
                loss_dict['lpipsp'] = lpipsp
                loss_prefix += lpipsp * self.hparams.lpips_prefix

            if self.hparams.adv_prefix > 0:
                axp = self.add_loss_adv(a=self.recon_prefix,
                                        net_d=self.net_d_prefix, truth=True)
                loss_dict['axp'] = axp
                loss_prefix += axp * self.hparams.adv * self.hparams.adv_prefix

            loss_dict['sum'] = loss_dict['sum'] + loss_prefix
        return loss_dict

    def backward_d(self):
        loss_dict = super().backward_d()
        if self.hparams.adv_prefix > 0 and self.recon_prefix is not None:
            # fakes = prefix decodes; reals = band-limited real slices, so the
            # discriminator scores "clean coarse image" rather than sharpness
            dp = self.add_loss_adv(a=self.recon_prefix, net_d=self.net_d_prefix, truth=False) \
                + self.add_loss_adv(a=self._prefix_target(), net_d=self.net_d_prefix, truth=True)
            loss_dict['dp'] = dp
            loss_dict['sum'] = loss_dict['sum'] + dp * self.hparams.adv * self.hparams.adv_prefix
        return loss_dict

    def configure_optimizers(self):
        # Same as MSclean, with net_d_prefix folded into opt_disc when present.
        lr_d = self.hparams.lr
        lr_g = getattr(self.hparams, 'lr_g_factor', 1.0) * self.hparams.lr

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

        disc_params = (list(self.loss.discriminator.parameters()) +
                       list(self.net_d.parameters()) +
                       list(self.net_d_128.parameters()) +
                       list(self.net_d_64.parameters()))
        if hasattr(self, 'net_d_prefix'):
            disc_params += list(self.net_d_prefix.parameters())
        opt_disc = torch.optim.Adam(disc_params, lr=lr_d, betas=(0.5, 0.9))

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
