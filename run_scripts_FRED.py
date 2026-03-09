import subprocess
import shlex
from datetime import datetime
import time
import os
import multiprocessing
import functools
import sys
from config import *
sys.path.append(WORKPATH)
experiment_file = 'exp_FRED.py'
script_path = os.path.dirname(__file__)
script_folder = os.path.basename(script_path)

def run_script(fred_idx, log_dir):
    for seed in [4867]:
        n_trials = 150
        log_dir_new = f"FRED/{log_dir}/fred_idx{fred_idx}_trial{n_trials}_seed{seed}"
        subprocess.call(
            shlex.split(f"python {script_path}/{experiment_file} --suffix 'experiments' --log_dir "
                        f"'{log_dir_new}'  --num_epoch '300' --n_trials '{n_trials}' --fred_idx '{fred_idx}' "
                    f"--seed '{seed}' --use_scheduler_step"))
        
if __name__ == "__main__":
    # subprocess.call(['python', "./far_exp.py --record_dir 'logs' --suffix 'test' --memo 'let us check this'"])
    multiprocess = True
    suffix = f'FRED_ALL{script_folder}'
    start_time = time.time()
    log_dir = datetime.fromtimestamp(start_time).strftime("%y%m%d-%H%M%S.%f") + suffix
    text = f'START {suffix} \n {__file__}'

    mylist = list(range(127))#
    p = multiprocessing.Pool(1)
    result = p.map(functools.partial(run_script, log_dir=log_dir), mylist)
    print('time taken:', time.time() - start_time)
    time_taken = time.time() - start_time
    text = f'END {suffix} \n {__file__} \
        \nTime taken is {time_taken//60} min'

    