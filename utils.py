import argparse
import os
import sys


def get_options():
    parser = argparse.ArgumentParser(description="Texture Synthesis Parameters")

    parser.add_argument('-t', '--tar_dir', type=str, required=True)
    parser.add_argument('-s', '--seed', type=int, default=0)
    parser.add_argument('-c', '--cuda_id', type=int, default=0)
    parser.add_argument('--stop_itr', type=int, default=-1)

    # dataset setting
    parser.add_argument('-r', '--data_root', type=str, default='data')
    parser.add_argument('-d', '--dataset', type=str, default='imagenet')
    parser.add_argument('-u', '--use_train', action='store_true')

    #l-bfgs parameters optimisation
    parser.add_argument('--max_iter', type=int, default=2000)
    parser.add_argument('--max_cor', type=int, default=20)

    # loss function
    parser.add_argument('--tex_layers', type=str, nargs='*',
                        default=['pool4', 'pool3', 'pool2', 'pool1', 'conv1_1'])
    parser.add_argument('--tex_weights', type=float, nargs='*',
                        default=[1e9,1e9,1e9,1e9,1e9])

    opt = parser.parse_args()
    print_options(parser, opt)
    return opt


def print_options(parser, opt):
    message = ''
    message += '---------------------------- Options --------------------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = '\t[default: {}]'.format(str(default))
        message += '{:>15}: {:<25}{}\n'.format(str(k), str(v), comment)
    message += '---------------------------- End ------------------------------'
    print(message)


def check_array(x, name):
    print("-" * 10, name, "-" * 10)
    print("dtype: {} / "
          "shape: {} / "
          "min: {:.4f} / "
          "mean: {:.4f} / "
          "max: {:.4f}".format(
              x.dtype, x.shape, x.min(), x.mean(), x.max()
              )
          )
    print("-" * 10 + "-" * (len(name)+2) + "-" * 10)


def check_dirs(target_root, dir_list):
    for d in dir_list:
        dirpath = os.path.join(target_root, d.split('/')[-1])
        os.makedirs(dirpath, exist_ok=True)
