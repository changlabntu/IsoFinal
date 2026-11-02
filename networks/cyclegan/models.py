import torch.nn as nn
import torch.nn.functional as F
import torch


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


##############################
#           RESNET
##############################


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        )

    def forward(self, x):
        return x + self.block(x)


class GeneratorResNet(nn.Module):
    def __init__(self, input_shape, num_residual_blocks):
        super(GeneratorResNet, self).__init__()

        channels = input_shape[0]

        # Initial convolution block
        out_features = 64
        model = [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(channels, out_features, 7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]
        in_features = out_features

        # Downsampling
        for _ in range(2):
            out_features *= 2
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Residual blocks
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(out_features)]

        # Upsampling
        for _ in range(2):
            out_features //= 2
            model += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Output layer
        model += [nn.ReflectionPad2d(channels), nn.Conv2d(out_features, channels, 7), nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


##############################
#        Discriminator
##############################


class BlurPool2d(nn.Module):
    """Anti-aliased downsampling (Zhang, ICML 2019): fixed binomial low-pass, then
    stride-2 subsample. Prevents near-Nyquist content (e.g. the period-2/4 alias
    lattice of ConvTranspose generators) from folding into a phase-dependent
    response the discriminator cannot learn from. No learnable parameters; the
    kernel buffer is non-persistent (kept out of checkpoints). Note: inserting
    blur modules shifts nn.Sequential indices, so a blur D's state_dict KEYS
    differ from a plain patch D's even though every parameter tensor keeps the
    same shape and order — warm-starting from a plain checkpoint needs a key
    remap, not strict loading."""

    def __init__(self, channels, stride=2):
        super().__init__()
        self.stride = stride
        k = torch.tensor([1., 2., 1.])
        k = (k[:, None] * k[None, :]) / 16.0  # 3x3 binomial, sums to 1
        self.register_buffer('kernel', k.expand(channels, 1, 3, 3).clone(),
                             persistent=False)

    def forward(self, x):
        return F.conv2d(x, self.kernel, stride=self.stride, padding=1,
                        groups=x.shape[1])


class Discriminator(nn.Module):
    def __init__(self, input_shape, patch, ndf=64, blur=False):
        super(Discriminator, self).__init__()
        assert patch in [4, 8, 16]
        print('Use ' + str(patch) + ' patch discriminator' + (' (blurpool)' if blur else ''))
        channels, height, width = input_shape

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block.

            blur=False: Conv(k4,s2) -> [InstanceNorm] -> LeakyReLU (original).
            blur=True (anti-aliased): Conv(k4,s1) -> [InstanceNorm] -> LeakyReLU
            -> BlurPool2d(s2) — blur goes last, after the nonlinearity, per Zhang;
            blurring before the activation would be defeated by the nonlinearity
            regenerating high frequencies. Output sizes match the s2 original
            (H -> H-1 -> H//2 for even H); conv weight shapes are identical
            (Sequential indices shift — see BlurPool2d docstring)."""
            stride = 1 if blur else 2
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=stride, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            if blur:
                layers.append(BlurPool2d(out_filters))
            return layers

        if 1:
            layers = [*discriminator_block(channels, ndf, normalize=False),
                      *discriminator_block(ndf, ndf*2)]

            if (patch == 4) or (patch == 8):
                layers = layers + [*discriminator_block(ndf*2, ndf*2)]

            layers = layers + [*discriminator_block(ndf*2, ndf*4)]

            if patch == 4:
                layers = layers + [*discriminator_block(ndf*4, ndf*4)]

            layers = layers + [*discriminator_block(ndf*4, ndf*8),
                    nn.ZeroPad2d((1, 0, 1, 0)),
                    nn.Conv2d(ndf*8, 1, 4, padding=1)]

            self.model = nn.Sequential(*layers)
        else:
            self.model = nn.Sequential(
                *discriminator_block(channels, 64, normalize=False),
                *discriminator_block(64, 128),
                *discriminator_block(128, 256),
                *discriminator_block(256, 512),
                nn.ZeroPad2d((1, 0, 1, 0)),
                nn.Conv2d(512, 1, 4, padding=1)
            )

    def forward(self, img):
        out = self.model(img)
        return out,


if __name__ == '__main__':
    d = Discriminator((3, 256, 256), patch=16)
    print(d(torch.rand((1, 3, 256, 256)))[0].shape)

