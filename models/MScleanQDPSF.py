from models.MScleanQD import GAN as MScleanQDGAN
import torch
import torch.nn.functional as F


# MScleanQDPSF: MScleanQD with a Z-consistency loss that replays the acquisition.
#
# Problem this fixes: the --l1how options of MSclean (max / mean / dsp over a
# block of output planes) are not how a slice is formed. With 'max' over 32
# planes one bright voxel anywhere in the block satisfies the loss, so the
# output is sparse and sits mid-block (about 14 planes too deep), and the true
# volume scores worse than a volume that was not deblurred at all
# (synthetic/README.md, stage 4; synthetic/sweep/loss_at_truth.py).
#
# Mechanism (--l1how psf, meant for --skipl1 1): the generator output is put
# through the measured acquisition and compared with every input slice.
#   1. undo the input mapping (nm '11g': floor --gamma_lo, exponent --gamma)
#      to get linear intensity, and read the floor as the background level
#      --psf_bg: the model cannot output below the floor, but the tissue
#      background lies below it, and a blur of floor-limited intensity would
#      put a false glow around every structure
#   2. blur with the measured 3D transfer (amplitude exp(-0.5 (|fz| / f0)^p),
#      f0 = (psf_a + psf_b * f_r) / psf_width; FFT with reflect padding)
#   3. take the slice positions: slice i sits at the centre of its planes
#      (mean of the two central planes), as in trilinear upsampling
#   4. redo the input mapping and compare with the input slices (L1), leaving
#      out --psf_margin slices at each end of the crop, where the blur would
#      need planes beyond the crop
# The coarse heads (out128 / out64) go through the same steps on their own
# grid, against the XY-pooled input, as in MSclean.
#
# To run without a floor (exact forward model, but the model then sees the
# background and its noise): --gamma_lo -1 --psf_bg -1.
#
# Expected output position: slice i at plane 8 i + 3.5 (not 8 i + 14).
#
# Any other --l1how makes this model identical to MScleanQD.
#
# NOTE (snapshot): train.py copies only this file into the checkpoint dir; it
# subclasses models/MScleanQD.py and models/MSclean.py.

DEFAULTS = dict(psf_a=0.00604, psf_b=0.04292, psf_p=1.3855,   # THX10 fit, synthetic/config/degradation.json
                psf_width=1.07, psf_bg=-0.9, psf_margin=2)
PAD_Z, PAD_XY = 32, 16     # reflect padding before the FFT, in full-resolution px
EPS = 0.01                 # fraction of the range above the floor below which the slope of the gamma mapping is held
LEAK = 0.05                # slope added to the inverse mapping in the backward pass (it is flat at the floor)


def psf_amplitude(shape, scale, a, b, p, width, device):
    """Amplitude transfer on the rfft grid of a (Y, X, Z) volume whose voxels
    are `scale` full-resolution px."""
    fy = torch.fft.fftfreq(shape[0], device=device)[:, None, None] / scale
    fx = torch.fft.fftfreq(shape[1], device=device)[None, :, None] / scale
    fz = torch.fft.rfftfreq(shape[2], device=device)[None, None, :] / scale
    f0 = (a + b * torch.sqrt(fy ** 2 + fx ** 2)) / width
    return torch.exp(-0.5 * (fz / f0) ** p)


def psf_slices(x, rate, scale, transfer, gamma, gamma_lo, bg, cache=None):
    """Generator output -> the slices the microscope would record from it.

    x: (B, C, Y, X, Z * rate) in model units, voxels of `scale` full-resolution
    px. transfer: (a, b, p, width). Returns (B, C, Y, X, Z) in model units.
    The values are exact; two slopes are changed in the backward pass only, so
    that voxels at the floor still receive a gradient and predictions just
    above the floor do not receive an unbounded one.
    """
    u = (x.float() + 1) / 2
    lin = u.clamp(min=0) ** (1 / gamma) + LEAK * (u - u.detach())      # 0 = floor, 1 = ceiling, linear in intensity
    v = bg + lin * (1 - bg)                                            # window units, floor read as background
    pz, pxy = PAD_Z // scale, PAD_XY // scale
    v = F.pad(v, (pz, pz, pxy, pxy, pxy, pxy), mode='reflect')
    key = (tuple(v.shape[2:]), scale, str(v.device)) + tuple(transfer)
    if cache is None or key not in cache:
        amp = psf_amplitude(v.shape[2:], scale, *transfer, v.device)
        if cache is not None:
            cache[key] = amp
    else:
        amp = cache[key]
    v = torch.fft.irfftn(torch.fft.rfftn(v, dim=(2, 3, 4)) * amp, s=v.shape[2:], dim=(2, 3, 4))
    v = v[:, :, pxy:v.shape[2] - pxy, pxy:v.shape[3] - pxy, pz:v.shape[4] - pz]
    s = 0.5 * (v[..., rate // 2 - 1::rate] + v[..., rate // 2::rate])  # slice i at the centre of its planes
    w = (s - gamma_lo) / (1 - gamma_lo)                                # fraction of the range above the floor
    soft = torch.where(w >= EPS, w.clamp(min=EPS) ** gamma, EPS ** gamma + gamma * EPS ** (gamma - 1) * (w - EPS))
    exact = w.detach().clamp(0, 1) ** gamma
    return 2 * (soft + (exact - soft).detach()) - 1


class GAN(MScleanQDGAN):
    def __init__(self, hparams, train_loader, eval_loader, checkpoints):
        MScleanQDGAN.__init__(self, hparams, train_loader, eval_loader, checkpoints)
        self._psf_cache = {}
        if self.hparams.l1how == 'psf' and self.hparams.nm != '11g':
            raise ValueError("--l1how psf inverts the '11g' input mapping; got --nm %s" % self.hparams.nm)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parent_parser = MScleanQDGAN.add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("MScleanQDPSF")
        parser.add_argument("--psf_a", type=float, default=DEFAULTS['psf_a'],
                            help='--l1how psf: axial cut-off of the transfer at zero in-plane frequency (cycles/px)')
        parser.add_argument("--psf_b", type=float, default=DEFAULTS['psf_b'],
                            help='--l1how psf: growth of the axial cut-off with in-plane frequency')
        parser.add_argument("--psf_p", type=float, default=DEFAULTS['psf_p'],
                            help='--l1how psf: exponent of the transfer')
        parser.add_argument("--psf_width", type=float, default=DEFAULTS['psf_width'],
                            help='--l1how psf: axial width of the blur relative to the fit (cut-off divided by it)')
        parser.add_argument("--psf_bg", type=float, default=DEFAULTS['psf_bg'],
                            help='--l1how psf: background level that the output floor stands for, in the '
                                 'units of --gamma_lo; equal to --gamma_lo takes the floor as it is')
        parser.add_argument("--psf_margin", type=int, default=DEFAULTS['psf_margin'],
                            help='--l1how psf: slices left out of the loss at each end of the crop')
        return parent_parser

    def _psf(self, name):
        return getattr(self.hparams, name, DEFAULTS[name])

    def get_projection(self, x, depth, how='mean', uprate=None):
        if how != 'psf':
            return super().get_projection(x, depth, how, uprate)
        h = self.hparams
        rate = self.uprate if uprate is None else uprate               # output planes per slice in this head
        scale = self.uprate // rate                                    # voxel size of this head in full-resolution px
        pred = psf_slices(x, rate, scale, tuple(self._psf(k) for k in ('psf_a', 'psf_b', 'psf_p', 'psf_width')),
                          h.gamma, h.gamma_lo, self._psf('psf_bg'), self._psf_cache)
        # backward_g takes the mean |difference| to the input slices over all
        # slices. The border slices are set to their target (no loss, no
        # gradient) and the rest is scaled up, so that mean equals the mean
        # over the slices that are used.
        target = self.oriX if scale == 1 else F.avg_pool3d(self.oriX, (scale, scale, 1))
        n = target.shape[-1]
        m = min(self._psf('psf_margin'), (n - 1) // 2)
        keep = torch.zeros(n, device=x.device)
        keep[m:n - m] = 1
        keep = keep[::h.skipl1]
        pred, target = pred[..., ::h.skipl1], target[..., ::h.skipl1]
        return target + (pred - target) * keep * (keep.numel() / keep.sum())
