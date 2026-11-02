from typing import Dict, List, Optional, Union, Any

import os
import struct
import time
from io import BytesIO
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import yaml
import torchvision.models as models
from pytorch_lightning.loggers import MLFlowLogger
from mlflow.exceptions import MlflowException
from taming.modules.losses.lpips import LPIPS
from torchmetrics.image.kid import KernelInceptionDistance
from networks.networks import get_scheduler
from networks.loss import GANLoss
from networks.registry import network_registry
from utils.metrics_spectral import mean_power_spectrum, lattice_peak_ratios, band_power
from utils.data_utils import *
import tifffile as tiff


def _weights_init(m: nn.Module) -> None:
    """Initialize network weights using normal distribution."""
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)


class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]

        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


class BaseModel(pl.LightningModule):
    """Base model for GAN training using PyTorch Lightning."""
    
    def __init__(self, hparams: Any, train_loader: Any, eval_loader: Any, checkpoints: str) -> None:
        super().__init__()
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.epoch = 0
        self.dir_checkpoints = checkpoints

        # Model and loss names
        self.netg_names = {'net_g': 'netG'}
        self.netd_names = {'net_d': 'netD'}
        self.loss_g_names = ['loss_g']
        self.loss_d_names = ['loss_d']

        # Process hyperparameters
        hparams_dict = vars(hparams).copy()
        for k in hparams.not_tracking_hparams:
            hparams_dict.pop(k, None)
        hparams_dict.pop("not_tracking_hparams", None)
        self.hparams.update(hparams_dict)
        self.save_hyperparameters(self.hparams)

        # Initialize loss functions
        self._init_loss_functions()

        # Final
        self.hparams.update(vars(self.hparams))   # updated hparams to be logged in tensorboard
        #self.train_loader.dataset.shuffle_images()  # !!! shuffle again just to make sure
        #self.train_loader.dataset.shuffle_images()  # !!! shuffle again just to make sure

        self.all_label = []
        self.all_out = []
        self.all_loss = []

        self.buffer = {}
        self._epoch_time_sum = 0.0
        self._epoch_count = 0
        self.best_val_loss = float('inf')
        self.best_epoch = -1
        os.makedirs('out', exist_ok=True)

    def _init_loss_functions(self) -> None:
        """Initialize loss functions."""
        self.criterionL1 = nn.L1Loss()
        self.criterionL2 = nn.MSELoss()
        if self.hparams.gan_mode == 'vanilla':
            self.criterionGAN = nn.BCEWithLogitsLoss()
        else:
            self.criterionGAN = GANLoss(self.hparams.gan_mode)
        self.lpips_fn = LPIPS().eval()
        for p in self.lpips_fn.parameters():
            p.requires_grad = False
        self.kid = KernelInceptionDistance(feature=64, subset_size=16)

    def _get_mlflow_client_and_run_id(self):
        """Get the MLflow client and run ID.

        Only returns valid references on rank 0 to avoid duplicate logging in DDP.

        Returns:
            Tuple of (MlflowClient, run_id), or (None, None) if unavailable.
        """
        if not hasattr(self, 'trainer') or self.trainer is None:
            return None, None
        if not self.trainer.is_global_zero:
            return None, None
        for logger in self.loggers:
            if isinstance(logger, MLFlowLogger):
                client = logger.experiment
                run_id = logger.run_id
                if client is not None and run_id is not None:
                    return client, run_id
        return None, None

    def on_fit_start(self):
        """Log config.json and YAML config to MLflow artifacts at the start of training."""
        client, run_id = self._get_mlflow_client_and_run_id()
        if client is None:
            return
        try:
            config_path = os.path.join(self.dir_checkpoints, 'config.json')
            client.log_artifact(run_id, config_path, artifact_path="config")
            yaml_path = os.path.join(self.dir_checkpoints, self.hparams.yaml + '.yaml')
            client.log_artifact(run_id, yaml_path, artifact_path="config")
        except (MlflowException, OSError) as e:
            print(f"WARNING: Failed to log config artifacts to MLflow: {e}")

    @staticmethod
    def _save_3d_as_gif(arr, path, duration=100):
        """Create an animated GIF by sweeping axis 0 of a 3D array (one frame per index).

        The file is assembled by hand from per-frame single-image GIF saves:
        every frame is complete and opaque, indexed against one shared
        256-gray global palette. Pillow's multi-frame writer (save_all) is
        avoided because it stores pixels unchanged since the previous frame
        as transparent "reuse previous" deltas, which spec-naive readers
        (ImageJ) render as black speckle on every frame after the first.

        Args:
            arr: 3D numpy array with shape (frames, H, W). Axis 0 is the sweep axis,
                 NOT necessarily the volume's Z: validation passes (Y, X, Z*panels),
                 so each frame is an XZ cut and the animation sweeps Y.
            path: Output file path for the GIF.
            duration: Duration per frame in milliseconds.
        """
        if arr.ndim != 3:
            print(f"WARNING: _save_3d_as_gif expected 3D array, got {arr.ndim}D — skipping")
            return
        arr = arr.astype(np.float32)
        mn, mx = arr.min(), arr.max()
        if mx - mn > 1e-8:
            arr = (arr - mn) / (mx - mn)
        else:
            arr = np.zeros_like(arr)
        arr_uint8 = (arr * 255).astype(np.uint8)

        # Deliberate: only the first half of the sweep is written (keeps the GIF small).
        n_frames = arr_uint8.shape[0] // 2
        if n_frames == 0:
            return
        h, w = arr_uint8.shape[1:]
        gray_ramp = bytes(v for g in range(256) for v in (g, g, g))

        def frame_image_data(slice_uint8):
            # Single-frame saves are never delta-encoded; splice out the
            # image descriptor + LZW data and re-verify the palette so the
            # spliced indices stay valid against the shared global ramp.
            frame = Image.frombytes('P', (w, h), slice_uint8.tobytes())
            frame.putpalette(gray_ramp)
            buf = BytesIO()
            frame.save(buf, format='GIF', optimize=False)
            data = buf.getvalue()
            flags = data[10]
            pal_len = 3 * (2 << (flags & 7)) if flags & 0x80 else 0
            if data[13:13 + pal_len] != gray_ramp:
                raise ValueError('Pillow rewrote the palette; frame indices would be remapped')
            pos = 13 + pal_len
            while data[pos] == 0x21:  # skip extension blocks
                pos += 2
                while data[pos] != 0:
                    pos += 1 + data[pos]
                pos += 1
            if data[pos] != 0x2C or data[pos + 9] & 0x80:
                raise ValueError('unexpected single-frame GIF layout')
            end = pos + 10 + 1  # image descriptor + LZW minimum code size
            while data[end] != 0:
                end += 1 + data[end]
            return data[pos:end + 1]

        delay_cs = max(1, duration // 10)
        gce = struct.pack('<BBBBHBB', 0x21, 0xF9, 4, 0, delay_cs, 0, 0)
        try:
            payload = [frame_image_data(arr_uint8[z]) for z in range(n_frames)]
        except ValueError as e:
            print(f"WARNING: manual GIF assembly failed ({e}) — falling back to Pillow save_all")
            frames = [Image.fromarray(arr_uint8[z]) for z in range(n_frames)]
            frames[0].save(
                path, save_all=True, append_images=frames[1:],
                duration=duration, loop=0
            )
            return
        with open(path, 'wb') as f:
            f.write(b'GIF89a')
            f.write(struct.pack('<HHBBB', w, h, 0xF7, 0, 0))
            f.write(gray_ramp)
            f.write(b'\x21\xFF\x0BNETSCAPE2.0\x03\x01\x00\x00\x00')  # loop forever
            for p in payload:
                f.write(gce)
                f.write(p)
            f.write(b'\x3B')

    def _log_gif_artifact(self, arr, prefix):
        """Create a GIF from a 3D array and log to MLflow artifacts.

        Args:
            arr: 3D numpy array with shape (frames, H, W). Axis 0 is the sweep axis,
                 NOT necessarily the volume's Z: validation passes (Y, X, Z*panels),
                 so each frame is an XZ cut and the animation sweeps Y.
            prefix: Filename prefix (e.g. 'train'), used as ``{prefix}_epoch_{N}.gif``.
        """
        client, run_id = self._get_mlflow_client_and_run_id()
        if client is None:
            return
        gif_path = os.path.join(self.dir_checkpoints, f'{prefix}_epoch_{self.epoch}.gif')
        try:
            self._save_3d_as_gif(arr, gif_path)
            if os.path.exists(gif_path):
                client.log_artifact(run_id, gif_path, artifact_path="images")
        except (MlflowException, OSError) as e:
            print(f"WARNING: Failed to log {prefix} GIF artifact to MLflow: {e}")
        finally:
            if os.path.exists(gif_path):
                os.remove(gif_path)

    def configure_optimizers(self):  ## THIS IS REQUIRED
        print('configuring optimizer being called....')
        print(self.netg_names.keys())
        print(self.netd_names.keys())
        print('configuring optimizer done')

        netg_parameters = []
        for g in self.netg_names.keys():
            netg_parameters = netg_parameters + list(getattr(self, g).parameters())
        print('Number of parameters in generator: ', sum(p.numel() for p in netg_parameters if p.requires_grad))

        netd_parameters = []
        for d in self.netd_names.keys():
            netd_parameters = netd_parameters + list(getattr(self, d).parameters())
        print('Number of parameters in discriminator: ', sum(p.numel() for p in netd_parameters if p.requires_grad))

        self.optimizer_g = optim.Adam(netg_parameters, lr=self.hparams.lr, betas=(self.hparams.beta1, 0.999))
        self.optimizer_d = optim.Adam(netd_parameters, lr=self.hparams.lr, betas=(self.hparams.beta1, 0.999))
        self.net_g_scheduler = get_scheduler(self.optimizer_g, self.hparams)
        self.net_d_scheduler = get_scheduler(self.optimizer_d, self.hparams)
        # not using pl scheduler for now....
        return [self.optimizer_g, self.optimizer_d], []

    def add_loss_adv(self, a, net_d, truth):
        disc_logits = net_d(a)[0]
        if truth:
            # Create a tensor of ones with the same shape as disc_logits
            target = torch.ones_like(disc_logits)
        else:
            # Create a tensor of zeros with the same shape as disc_logits
            target = torch.zeros_like(disc_logits)
        
        adv = self.criterionGAN(disc_logits, target)
        return adv

    def add_loss_l1(self, a, b):
        l1 = self.criterionL1(a, b)
        return l1

    def add_loss_l2(self, a, b):
        l1 = self.criterionL2(a, b)
        return l1

    def training_step(self, batch, batch_idx, optimizer_idx):
        if optimizer_idx == 1:
            self.generation(batch)
            loss_d = self.backward_d()
            if loss_d is not None:
                for k in list(loss_d.keys()):
                    if k != 'sum':
                        self.log(k, loss_d[k], on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
                return loss_d['sum']
            else:
                return None

        if optimizer_idx == 0:
            self.generation(batch)
            loss_g = self.backward_g()
            for k in list(loss_g.keys()):
                if k != 'sum':
                    self.log(k, loss_g[k], on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            
            return loss_g['sum']

    def on_train_epoch_start(self):
        """Records epoch start time for duration tracking."""
        self._epoch_start_time = time.time()

    def training_epoch_end(self, outputs):
        if hasattr(self, '_epoch_start_time'):
            epoch_duration = time.time() - self._epoch_start_time
            self._epoch_time_sum += epoch_duration
            self._epoch_count += 1
            avg = self._epoch_time_sum / self._epoch_count
            if self.trainer.is_global_zero:
                print(f'Epoch {self.epoch} | {epoch_duration//60:.0f}m {epoch_duration%60:.1f}s (avg: {avg//60:.0f}m {avg%60:.1f}s)')
            client, run_id = self._get_mlflow_client_and_run_id()
            if client is not None:
                client.log_metric(run_id, 'epoch_time', epoch_duration, step=self.epoch)

        self.log('lr_g', self.trainer.optimizers[0].param_groups[0]['lr'],
                 on_step=False, on_epoch=True, logger=True, sync_dist=True)

        # checkpoint — rank 0 only to avoid duplicate writes
        if self.epoch % self.hparams.epoch_save == 0 and self.trainer.is_global_zero:
            for name in self.netg_names.keys():
                path_g = os.path.join(self.dir_checkpoints, '{}_model_epoch_{}.pth'.format(self.netg_names[name], self.epoch))
                torch.save(getattr(self, name), path_g)
                print("Checkpoint saved to {}".format(path_g))

            if self.hparams.save_d:
                for name in self.netd_names.keys():
                    path_d = os.path.join(self.dir_checkpoints, '{}_model_epoch_{}.pth'.format(self.netd_names[name], self.epoch))
                    torch.save(getattr(self, name), path_d)
                    print("Checkpoint saved to {}".format(path_d))

        self.net_g_scheduler.step()
        self.net_d_scheduler.step()

        self.reset_metrics()

        # GIF artifact is logged from validation's first batch in validation_epoch_end

        self.epoch += 1

    def get_metrics(self):
        pass

    def reset_metrics(self):
        pass

    def validation_step(self, batch, batch_idx):
        if not hasattr(self, 'generation'):
            return None
        self.generation(batch)
        if not (hasattr(self, 'Xup') and hasattr(self, 'XupX')):
            return None

        # Store first batch for artifact logging (rank 0 only).
        # Order for the GIF: data0 (input) -> enhanced -> data1 -> data2 -> ...
        # Extra views (xcube/ycube/...) are interpolated to the output shape so the
        # panels concatenate cleanly; single-view runs just give [input, enhanced].
        if batch_idx == 0 and self.trainer.is_global_zero:
            views = [self.Xup.detach().clone(), self.XupX.detach().clone()]
            for v in getattr(self, 'aux_views', []):
                v = nn.functional.interpolate(v, size=self.XupX.shape[2:], mode='trilinear')
                views.append(v.detach().clone())
            # Coarse multi-scale outputs (e.g. vqcleanM0aMS's 1/2 and 1/4 heads). Upsample
            # to the full-res grid with 'trilinear' so the panels concatenate cleanly and
            # render as smooth lower-res copies of the full-res output. ('nearest' was tried
            # to make the resolution difference obvious, but it blows each native voxel into
            # a hard block that reads as a checkerboard artifact.) Guarded — models without
            # gif_scales are unaffected and keep the original two-panel layout.
            for v in getattr(self, 'gif_scales', []):
                v = nn.functional.interpolate(v, size=self.XupX.shape[2:], mode='trilinear')
                views.append(v.detach().clone())
            self.val_views = views

        def to_rgb(t):
            return t.repeat(1, 3, 1, 1) if t.shape[1] == 1 else t

        with torch.no_grad():
            # NOTE: the local names X and Y below are SWAPPED relative to the repo
            # convention (B, C, Y, X, Z): here `X` is dim2 (= Y) and `Y` is dim3 (= X).
            # Consequently `yz_*` holds XZ planes and `xz_pred` holds YZ planes. The
            # index arithmetic is correct; only the names mislead. Left as-is on purpose.
            B, C, X, Y, Z = self.XupX.shape
            # XY view: all Z-slices → (B*Z, C, Y, X)  (an XY plane per Z index)
            xy_tgt  = self.Xup.permute(0, 4, 1, 2, 3).reshape(B * Z, C, X, Y)[::16]
            # XZ view (despite the yz_ names): all Y-slices → (B*Y, C, X, Z)
            # [requires Y==Z for the LPIPS batch match]
            yz_tgt  = self.Xup.permute(0, 2, 1, 3, 4).reshape(B * X, C, Y, Z)[::16]
            yz_pred = self.XupX.permute(0, 2, 1, 3, 4).reshape(B * X, C, Y, Z)[::16]
            # lpips_tgt: XY vs XZ of ground truth (baseline anisotropy)
            # lpips_pred: XY of target vs XZ of prediction (isotropy of output)
            lpips_tgt  = self.lpips_fn(to_rgb(xy_tgt), to_rgb(yz_tgt)).mean()
            lpips_pred = self.lpips_fn(to_rgb(xy_tgt), to_rgb(yz_pred)).mean()

            # Lattice alias-peak metric: accumulate the mean power spectrum of both
            # Z-containing plane orientations (XZ = yz_pred, YZ = xz_pred) of the
            # prediction across val batches
            # (Welch averaging — stabler than averaging per-batch ratios); the
            # peak/background ratios are computed once in validation_epoch_end.
            # Keyed by spectrum shape so non-cubic volumes never mix sizes.
            # Init at batch_idx == 0 (NOT in on_validation_start — model subclasses
            # override that hook without calling super()).
            if batch_idx == 0:
                self._val_spec_acc = {}
            # YZ view (despite the name): all X-slices → (B*X, C, Y, Z)
            xz_pred = self.XupX.permute(0, 3, 1, 2, 4).reshape(B * Y, C, X, Z)[::16]
            for planes in (yz_pred, xz_pred):
                spec = mean_power_spectrum(planes.float())
                k = tuple(spec.shape)
                if k in self._val_spec_acc:
                    self._val_spec_acc[k][0] += spec
                    self._val_spec_acc[k][1] += 1
                else:
                    self._val_spec_acc[k] = [spec, 1]

            # Mid/high-band spectral retention vs the observed input slices
            # (visual-parity tracking — doc/MS_sharpness.md, memory 2026-07-09):
            #   val_spec_recon_{mid,hi} — 2D VQ recon vs input (does the latent carry it?)
            #   val_spec_xy_{mid,hi}    — 3D output's XY planes vs input (does the trunk render it?)
            # Same Welch accumulation as the lattice metric; ratios in validation_epoch_end.
            # ~1 = full retention. References through this code path: see run.sh comments.
            if batch_idx == 0:
                self._val_band_acc = {}
            if hasattr(self, 'reconstructions') and hasattr(self, 'oriX'):
                inp_xy = self.oriX.permute(4, 1, 2, 3, 0)[:, :1, :, :, 0]      # (Zin, 1, Y, X)
                rec_xy = self.reconstructions[:, :1]                            # (B*Zin, 1, Y, X); B == 1 for the val loader
                step = max(1, int(getattr(self, 'uprate', 1)))
                pred_xy = self.XupX.permute(0, 4, 1, 2, 3).reshape(B * Z, C, X, Y)[::step, :1]  # (B*Z, C, Y, X) XY planes
                for name, planes in (('inp', inp_xy), ('rec', rec_xy), ('xy', pred_xy)):
                    if planes.shape[-2:] != inp_xy.shape[-2:]:
                        continue  # never mix plane sizes within a key
                    spec = mean_power_spectrum(planes.float())
                    if name in self._val_band_acc:
                        self._val_band_acc[name][0] += spec
                        self._val_band_acc[name][1] += 1
                    else:
                        self._val_band_acc[name] = [spec, 1]

        # Accumulate for the end-of-epoch console printout (this rank's shard only —
        # the logged metrics below are the properly rank-synced values).
        if batch_idx == 0:
            self._val_lpips_acc = {'tgt': [], 'pred': []}
        self._val_lpips_acc['tgt'].append(float(lpips_tgt))
        self._val_lpips_acc['pred'].append(float(lpips_pred))

        # on_epoch=True: PL averages across all batches and ranks automatically
        self.log('val_lpips_tgt', lpips_tgt, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log('val_lpips_pred', lpips_pred, on_step=False, on_epoch=True, logger=True, sync_dist=True)

        # Held-out real X/Y projection error (vqcleanM0aSup0-style). Mirrors backward_g's
        # l1_x/l1_y: pool XupX along dim3 (->xcube) and dim2 (->ycube) by --aniso (op =
        # --l1how_xy) and L1-match the real views. cropz is train-only, so at val the
        # views' Z can differ from XupX by subsample rounding -> interpolate to XupX shape
        # first (as the GIF does). Guarded so models without lamb_xy/get_projection are
        # unaffected.
        if (hasattr(self, 'get_projection') and getattr(self.hparams, 'lamb_xy', 0) > 0
                and hasattr(self.hparams, 'aniso')
                and len(getattr(self, 'aux_views', [])) >= 2):
            r = self.hparams.aniso
            how_xy = getattr(self.hparams, 'l1how_xy', 'max')
            with torch.no_grad():
                xcube = nn.functional.interpolate(self.aux_views[0], size=self.XupX.shape[2:],
                                                  mode='trilinear')
                ycube = nn.functional.interpolate(self.aux_views[1], size=self.XupX.shape[2:],
                                                  mode='trilinear')
                val_l1_x = self.add_loss_l1(
                    a=self.get_projection(self.XupX, depth=r, how=how_xy, axis=3),
                    b=self.get_projection(xcube, depth=r, how=how_xy, axis=3))
                val_l1_y = self.add_loss_l1(
                    a=self.get_projection(self.XupX, depth=r, how=how_xy, axis=2),
                    b=self.get_projection(ycube, depth=r, how=how_xy, axis=2))
            self.log('val_l1_x', val_l1_x, on_step=False, on_epoch=True, logger=True, sync_dist=True)
            self.log('val_l1_y', val_l1_y, on_step=False, on_epoch=True, logger=True, sync_dist=True)

        # KID: accumulate XY(tgt) vs XZ(pred, the `yz_pred` stack) across all val batches
        # KernelInceptionDistance expects uint8 [0, 255]
        def to_uint8(t):
            t = to_rgb(t).clamp(-1, 1)
            return ((t + 1) / 2 * 255).to(torch.uint8)

        self.kid.update(to_uint8(xy_tgt), real=True)
        self.kid.update(to_uint8(yz_pred), real=False)
        return None

    def testing_step(self, batch, batch_idx):
        self.generation(batch)

    def validation_epoch_end(self, x):
        kid_mean, kid_std = self.kid.compute()
        self.log('val_kid', kid_mean, logger=True, sync_dist=True)
        self.kid.reset()

        # Lattice alias-peak ratios from the epoch-accumulated spectra (see
        # utils/metrics_spectral.py; ~1 = clean, the ConvTranspose lattice showed
        # p2diag up to ~90x). sync_dist averages the per-rank ratios — assumes
        # every rank saw >= 1 val batch (DistributedSampler pads; same assumption
        # val_kid makes above).
        spec_acc = getattr(self, '_val_spec_acc', None)
        lat = {}
        if spec_acc:
            for spec_sum, n in spec_acc.values():
                for k, v in lattice_peak_ratios(spec_sum / n).items():
                    lat.setdefault(k, []).append(v)
            for k in lat:
                lat[k] = torch.stack(lat[k]).mean()
                self.log('val_lat_' + k, lat[k], logger=True, sync_dist=True)
            self._val_spec_acc = {}

        # Spectral retention ratios vs the input's XY spectrum (see validation_step).
        # sync_dist averages the per-rank ratios (same assumption as val_lat_*).
        band_acc = getattr(self, '_val_band_acc', None)
        spec_ret = {}
        if band_acc and 'inp' in band_acc:
            inp_bands = band_power(band_acc['inp'][0] / band_acc['inp'][1])
            for name, tag in (('rec', 'recon'), ('xy', 'xy')):
                if name in band_acc:
                    b = band_power(band_acc[name][0] / band_acc[name][1])
                    for band in b:
                        ratio = b[band] / inp_bands[band].clamp_min(1e-12)
                        spec_ret[f'{tag}_{band}'] = ratio
                        self.log(f'val_spec_{tag}_{band}', ratio, logger=True, sync_dist=True)
            self._val_band_acc = {}

        # Console summary per epoch (rank 0's shard; MLflow/TB hold the synced values)
        acc = getattr(self, '_val_lpips_acc', None)
        if self.trainer.is_global_zero and acc and acc['pred']:
            msg = 'Epoch {} val | lpips_pred {:.4f} | lpips_tgt {:.4f} | kid {:.3f}'.format(
                self.epoch, np.mean(acc['pred']), np.mean(acc['tgt']), float(kid_mean))
            if lat:
                msg += ' | lat ' + ' '.join('{} {:.1f}'.format(k, float(v))
                                            for k, v in lat.items())
            if spec_ret:
                msg += ' | spec ' + ' '.join('{} {:.2f}'.format(k, float(v))
                                             for k, v in spec_ret.items())
            print(msg)

        # Log a GIF from validation's first batch (stored in validation_step), every 5 epochs.
        # Panels are laid out left-to-right as data0 (input) | enhanced | data1 | data2 | ...
        # Each frame is an XZ cut (X down, Z across); the animation sweeps Y (first half
        # only, see _save_3d_as_gif). This IS the side view.
        if (self.epoch % 5 == 0 and self.trainer.is_global_zero
                and hasattr(self, 'val_views') and len(self.val_views) >= 2):
            def to_panel(t):
                # (B, C, Y, X, Z) -> (Y, C*X, Z): channels stacked vertically, Z is the frame
                # width; .squeeze() assumes B == 1 (true for the val loader)
                return np.concatenate([t[:, c, ::].squeeze().detach().cpu().numpy()
                                       for c in range(t.shape[1])], 1)
            panels = [to_panel(v) for v in self.val_views]
            concat_arr = np.concatenate(panels, 2)  # side by side along Z (frame width)
            self._log_gif_artifact(concat_arr, 'val')

        return None

    def get_progress_bar_dict(self):
        tqdm_dict = super().get_progress_bar_dict()
        if 'v_num' in tqdm_dict:
            del tqdm_dict['v_num']
        if 'loss' in tqdm_dict:
            del tqdm_dict['loss']
        return tqdm_dict

    def generation(self, batch):
        pass

    def backward_g(self):
        pass

    def backward_d(self):
        pass

    def set_networks(self, net='all'):
        """Create network instances using the network registry.
        
        Args:
            net: Which networks to return ('all', 'g', or 'd')
            
        Returns:
            Generator and/or discriminator networks based on the net parameter
        """
        # Common parameters for both networks
        common_params = {
            'input_nc': self.hparams.input_nc,
            'output_nc': self.hparams.output_nc,
            'norm': self.hparams.norm,
            'norm_layer': nn.BatchNorm2d
        }
        
        # Generator-specific parameters
        generator_params = {
            **common_params,
            'ngf': self.hparams.ngf,
            'use_dropout': self.hparams.mc,
            'final': self.hparams.final,
            'mc': self.hparams.mc
        }
        
        # Discriminator-specific parameters
        discriminator_params = {
            **common_params,
            'ndf': self.hparams.ndf
        }
        
        # Create networks
        net_g = network_registry.get_generator(self.hparams.netG, **generator_params)
        net_d = network_registry.get_discriminator(self.hparams.netD, **discriminator_params)
        
        # Return requested networks
        if net == 'all':
            return net_g, net_d
        elif net == 'g':
            return net_g
        elif net == 'd':
            return net_d

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad


