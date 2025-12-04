# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 09:19:02 2025

Step 2.2.2: Identification / Summarize Runtime

Summarize the runtime and complexity of the indentification sessions of one model.  

@author: Christoph M. Konrad
"""
import os
import numpy as np
import pickle as pkl

from datetime import datetime

from rcid.utils import read_yaml, get_default_parser
from rcid.path_manager import PathManager

def to_hhmmss(dt):
    total_s = dt.total_seconds()
    h = total_s // 3600
    m = (total_s - h * 3600)//60
    s = total_s - h * 3600 - m * 60
    
    return f"{int(h):0>2}:{int(m):0>2}:{int(s):0>2}"

def count_results(dir_results):
    count = 0
    # count identification results by counting sol_XX.pkl files.
    for root, _, files in os.walk(dir_results):
        for file in files:
            if file.startswith("sol_") and file.lower().endswith(".pkl"):
                count += 1
    return count


def get_average_problem_complexity(dir_results):
    count = 0
    n_nodes=0
    n_states=0
    n_free=0
    # count identification results by counting traj_XX.png files.
    for root, dir, files in os.walk(dir_results):
        for file in files:
            if file.startswith("sol_") and file.lower().endswith(".pkl"):
                
                count += 1

                #print(dir)
                filepath = os.path.join(root, file)
                
                #pattern = r"sol_(\d{1,2})-(\d{1,2}).pkl"

                with open(filepath, 'rb') as f:
                    data = pkl.load(f)

                #print(data['solution'].keys())
                n_nodes += data['n_data']
                n_states += data['n_states']
                n_free += data['n_data'] * data['n_states'] + len(data['solution']['gains'])


    return count, n_nodes/count, n_states/count, n_free/count

def find_runtime(dir_results):
    
    timestamp_format = '%Y%m%d_%H%M%S'
    
    t_begin = []
    t_end = []
    runtimes = []
    subdirs = []
    msg = "Runtimes per identification session:\n"
    
    for root, subdir, files in os.walk(dir_results):
        for file in files:
            if file.lower().endswith(".log"):
                if not "eval-summary" in file:
                
                    with open(os.path.join(root, file), 'r') as f:
                        lines = f.readlines()
                    
                    lines = [l for l in lines if l.strip()]
                        
                    try:
                        t_begin_i = datetime.strptime(lines[0][:15], timestamp_format)
                        t_end_i = datetime.strptime(lines[-1][:15], timestamp_format)
                    except ValueError:
                        continue
                    
                    t_begin.append(t_begin_i)
                    t_end.append(t_end_i)
                    
                    
                    
                    runtimes.append(t_end_i - t_begin_i)
                    runtimestr = to_hhmmss(runtimes[-1])
                    
                    name = os.path.normpath(root)
                    name = name.split(os.sep)[-1]
                    
                    subdirs.append(name)
                    
                    msg += f" - {name}: {runtimestr} hrs.\n"
                    
    runtime = np.max(t_end) - np.min(t_begin)
    
    return t_begin, t_end, runtime, (runtimes, subdirs), msg
    
            
def parse_input():
    parser = get_default_parser(plot=False, model=True, program_info="Summarize the runtime and number of ID sessions for one model.")
    return parser.parse_args()
            
def main():

    args = parse_input()
    config = read_yaml(args.config)

    paths = PathManager(config['dir_data'], config['dir_results'])
    dir_results = config['dir_results']#paths.getdir_id(args.model)
    
    n_ids, n_nodes, n_states, n_free = get_average_problem_complexity(dir_results)
    t_begin, t_end, runtime, single, msg = find_runtime(dir_results)
    
    total_s = runtime.total_seconds()
    total_m = total_s / 60
    
    runtimestr = to_hhmmss(runtime)
    
    # make result message
    msg += f"Ran {n_ids} individual control identifications in {runtimestr} hrs.\n"
    msg += f"Average complexity: n_nodes={n_nodes:.1f}, n_states={n_states:.1f}, n_free={n_free:.1f}\n"
    msg += f"Identification rate is: {n_ids/total_m:.2f} identifications per minute.\n"

    id_fastest = np.argmin(single[0])
    msg += f"Fastest was {single[1][id_fastest]} with {to_hhmmss(single[0][id_fastest])} hrs.\n"
    
    id_slowest = np.argmax(single[0])
    msg += f"Slowest was {single[1][id_slowest]} with {to_hhmmss(single[0][id_slowest])} hrs.\n"

    # print and save
    print(msg)
    if args.save:
        with open(os.path.join(dir_results, f'rcid_runtime-summary.txt'), 'w') as f:
            f.write(msg)

if __name__ == "__main__":
    main()