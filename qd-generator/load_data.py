from mapf_qd import Individual, Container
import pickle
import networkx as nx
import tqdm
import math
import ruamel.yaml as yaml
import os

very_close = []
lam_value = 9e-4
os.mkdir(f'yaml_{lam_value}')
with open('20231005Oct1696539209lam_0.0009000000000000001/iteration_60000_container.pkl', 'rb') as f:
    data = pickle.load(f)
    for ind in tqdm.tqdm(data.individuals):
        g = ind.graph
        l2 = ind.lam2_value
        # if L2 of graph is very close to desired L2, add to list
        if abs(l2 - lam_value) < 1e-6:
            very_close.append(ind.obstacles)

            with open(f'yaml_{lam_value}/'+f'{ind.features[0]}_{ind.features[1]}'.replace('.', '-')+'.yaml', 'w') as g:
                ind.obstacles = [list(o) for o in ind.obstacles]
                yaml.dump({'dimensions': [32, 32], 'obstacles': ind.obstacles}, g)