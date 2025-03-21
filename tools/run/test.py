# Copyright (c) CAIRI AI Lab. All rights reserved

# This file was modified by multiple contributors.
# It includes code originally from the OpenSTL project (https://github.com/chengtan9907/OpenSTL).
# The original code is licensed under the Apache License, Version 2.0.

import os.path as osp
import time
import warnings
warnings.filterwarnings('ignore')

from simvp_minimal.experiment_recorder import generate_unique_id
from simvp_minimal.experiment import Experiment
from simvp_minimal.utils import create_parser, get_dist_info, generate_config, update_config, load_config, setup_multi_processes, format_seconds


if __name__ == '__main__':
    start_time = time.time()

    args = create_parser().parse_args()

    training_config = generate_config(args)

    config = args.__dict__
    config.update(training_config)


    if args.config_file is None:
        args.config_file = osp.join('./configs', args.dataname, f'{args.method}.py')

    config = update_config(config, load_config(args.config_file),
                           exclude_keys=['method'])

    if not config.get('ex_name'):
        config['ex_name'] = generate_unique_id(config)

    if not config.get('ex_dir'):
        config['ex_dir'] = f'{config["ex_name"]}'

    if not config.get('datafile_in'):
        raise ValueError('datafile_in is required')

    exp = Experiment(args)
    rank, world_size = get_dist_info()

    setup_multi_processes(config)

    if args.dist:
        print(f"Dist info: rank={rank}, world_size={world_size}")

    if config['inference'] and not config['test']:
        print('>' * 35 + f' inferencing {args.ex_name}  ' + '<' * 35)
        eval_res, _ = exp.inference()
    else:
        print('>' * 35 + f' testing {args.ex_name}  ' + '<' * 35)
        eval_res, _ = exp.test()

    if rank == 0:
        elapsed_time = time.time() - start_time
        print(f'Total time: {format_seconds(elapsed_time)}')