import argparse


def get_args():
    # Arguments
    parser = argparse.ArgumentParser()
    # Env
    parser.add_argument('--yaml', type=str, default='default', help='name of yaml config file')
    parser.add_argument('--env', type=str, default=None, help='environment_to_use')

    # Project name
    parser.add_argument('--prj', type=str, help='name of the project')
    parser.add_argument('--models', dest='models', type=str, help='use which models')

    # Data
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--preload', action='store_true')

    parser.add_argument('--precrop', type=int, help='size for precrop', default=0)
    parser.add_argument('--resize', type=int, help='size for resizing before cropping, 0 for no resizing')
    parser.add_argument('--cropsize', type=int, help='size for xy cropping, 0 for no crop')
    parser.add_argument('--cropz', type=int, help='size for z cropping, 0 for no crop')
    parser.add_argument('--rotate', action='store_true')

    parser.add_argument('--direction', type=str, help='paired: a_b, unpaired a%b ex:(a_b%c_d)')
    parser.add_argument('--nm', type=str, help='way to normalize itensity value')
    parser.add_argument('--gamma', type=float, default=0.25,
                        help="exponent for nm='11g': [-1,1] input is unit-mapped, raised to this power, "
                             "rescaled back to [-1,1]; <1 expands the dark range")
    parser.add_argument('--gamma_lo', type=float, default=-1.0,
                        help="floor for nm='11g': input at or below this maps to -1 (clips the noise "
                             "band), gamma then acts on [gamma_lo, 1] only; -1 disables the floor")

    # Model
    parser.add_argument('--gan_mode', type=str, help='gan mode')
    parser.add_argument('--netG', type=str, help='netG model')
    parser.add_argument('--norm', type=str, help='normalization in generator')
    parser.add_argument('--mc', action='store_true', dest='mc', default=False,
                        help='monte carlo dropout for some of the generators')
    parser.add_argument('--netD', type=str, help='netD model')
    parser.add_argument('--input_nc', type=int, help='input image channels')
    parser.add_argument('--output_nc', type=int, help='output image channels')
    parser.add_argument('--ngf', type=int, help='generator filters in first conv layer')
    parser.add_argument('--ndf', type=int, help='discriminator filters in first conv layer')

    parser.add_argument('--final', type=str, dest='final', help='activation of final layer')
    parser.add_argument('--trd', type=float, dest='trd', help='threshold of images')

    # Training
    parser.add_argument('--adv', type=float, default=1, help='weight for adversarial loss, 0 for no generator training')
    parser.add_argument('-b', dest='batch_size', type=int, help='training batch size')
    parser.add_argument('--n_epochs', type=int, help='# of iter at starting learning rate')
    parser.add_argument('--lr', type=float, help='initial learning rate f -or adam')
    parser.add_argument('--beta1', type=float, help='beta1 for adam. default=0.5')

    parser.add_argument('--epoch_save', type=int, default=20, help='to save checkpoint every epoch')
    parser.add_argument('--n_epochs_decay', type=int, help='# of iter to linearly decay learning rate to zero')
    parser.add_argument('--lr_policy', type=str, help='learning rate policy: lambda|step|plateau|cosine')
    parser.add_argument('--lr_decay_iters', type=int, help='multiply by a gamma every lr_decay_iters iterations')
    parser.add_argument('--save_d', action='store_true', dest='save_d', default=False,
                        help='save checkpoints of discriminators')

    # Loss
    parser.add_argument('--lamb', type=float, help='weight on L1 term in objective')
    # Misc
    parser.add_argument('--seed', type=int, help='random seed to use. Default=123')
    parser.add_argument('--mode', type=str, default='dummy')
    parser.add_argument('--port', type=str, default='dummy')
    parser.add_argument('--host', type=str, default='dummy')
    parser.add_argument('--tracking_uri', type=str, default=None,
                        help='MLflow tracking: an http://URL for a server, or a bare NAME for a '
                             'local store at $LOGS/NAME.db + $LOGS/NAME/ (default name: mlflow). '
                             'Overrides env config.')
    return parser
