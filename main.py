import argparse
import pdb

import yaml

from easyfl.distributed import slurm
from groups.coordinator_fssl import Coordinator_fssl

"""
config = {
    "data": {"dataset": "cifar10",
             "is_ssl": True,
             "ssl_senario": "client_part",
             "num_labels_per_class": 5
    }
}
"""


def get_args_parser():
    parser = argparse.ArgumentParser('FSSL', add_help=False)
    parser.add_argument('--config', default='', type=str)
    parser.add_argument('--output', default='', type=str)
    parser.add_argument('--port', default=23344, type=int)
    return parser


parser = get_args_parser()


def main():
    global args
    args = parser.parse_args()
    rank, local_rank, world_size, host_addr = slurm.setup(args.port)

    config_dis = {
        "gpu": world_size,
        "distributed": {
            "rank": rank,
            "local_rank": local_rank,
            "world_size": world_size,
            "init_method": host_addr,
            "backend": "nccl",
        },
    }

    with open(args.config) as f:
        config = yaml.full_load(f)
    # output_dir=args.output
    for k, v in config.items():
        if k in config_dis.keys():
            config[k] = config_dis[k]
    config.update(config_dis)
    config['output'] = args.output
    Group = Coordinator_fssl()

    Group.init(config, init_all=True)
    Group.run()


if __name__ == '__main__':
    main()
