import pandas as pd
from pybnesian.learning.algorithms import PC
from pybnesian.learning.independences import LinearCorrelation, RCoT
import pathlib
import os
import experiments_helper
import math
import multiprocessing as mp

def run_pc_graph(d, idx_dataset, i):
    pc = PC()

    for i in experiments_helper.INSTANCES:
        df = pd.read_csv('data/' + d + "_" + str(idx_dataset).zfill(3) + "_" + str(i) + '.csv')
        # df = pd.read_csv(d + "_"  + str(i) + '.csv')
        result_folder = 'models/' + d + '/' + str(idx_dataset).zfill(3) + '/' +  str(i) + '/PC/'
        pathlib.Path(result_folder).mkdir(parents=True, exist_ok=True)
    
        if not os.path.exists(result_folder + '/end-lc.lock'):
            lc = LinearCorrelation(df)
            
            graph_lc = pc.estimate(lc)
            graph_lc.save(result_folder + '/graph-lc')

            with open(result_folder + '/end-lc.lock', 'w') as f:
                pass
    
        if not os.path.exists(result_folder + '/end-rcot.lock'):
            rcot = RCoT(df)
            
            graph_rcot = pc.estimate(rcot, verbose=True)
            graph_rcot.save(result_folder + '/graph-rcot')

            with open(result_folder + '/end-rcot.lock', 'w') as f:
                pass

for d in experiments_helper.DATASETS:
    for i in experiments_helper.INSTANCES:
        for idx_dataset in range(0, math.ceil(experiments_helper.NUM_SIMULATIONS / 10)):
            
            num_processes = min(10, experiments_helper.NUM_SIMULATIONS - idx_dataset*10)
            with mp.Pool(processes=num_processes) as p:
                p.starmap(run_pc_graph, [(d, 10*idx_dataset + ii, i) for ii in range(num_processes)]
                        )