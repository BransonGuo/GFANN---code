import subprocess
import shlex
from datetime import datetime
import time
import os
import numpy as np
import multiprocessing
import os
import functools
import sys
from config import *
sys.path.append(WORKPATH)

experiment_file = "exp_simulation.py"
script_path = os.path.dirname(__file__)
script_folder = os.path.basename(script_path)


def run_script(seed, log_dir):
    y_func_l = list(np.random.randint(0, 8, 10))
    n_trials = 20
    for p in [500]:
        for factor_id in [1]:
            for hcm_id in [2]:
                log_dir_new = f"Simulation/{log_dir}trial_{n_trials}/p_{p}/seed{seed}"
                subprocess.call(
                    shlex.split(f"python {script_path}/{experiment_file} --suffix 'experiments' --log_dir "
                                f"'{log_dir_new}'  --p '{p}' --num_epoch '300' --n_trials '{n_trials}' --y_func_l '{y_func_l}' "
                            f"--factor_id '{factor_id}' --hcm_id '{hcm_id}' --seed '{seed}'"))

if __name__ == "__main__":
    # subprocess.call(['python', "./far_exp.py --record_dir 'logs' --suffix 'test' --memo 'let us check this'"])
    multiprocess = True
    suffix = f'Simluation_AllDim{script_folder}'
    start_time = time.time()
    log_dir = datetime.fromtimestamp(start_time).strftime("%y%m%d-%H%M%S.%f") + suffix
    text = f'START {suffix} \n {__file__}'

    if multiprocess:
        seeds = [140]#range(100, 200, 5)
        # if use GPU, just set 1
        p = multiprocessing.Pool(1)
        result = p.map(functools.partial(run_script, log_dir=log_dir), seeds)
        
    print('time taken:', time.time() - start_time)
    time_taken = time.time() - start_time
    text = f'END {suffix} \n {__file__} \
        \nTime taken is {time_taken//60} min'

    