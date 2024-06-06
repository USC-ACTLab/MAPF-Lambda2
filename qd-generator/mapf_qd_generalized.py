import os
import pickle
import random
from matplotlib import pyplot as plt
import networkx as nx
import copy
import numpy as np
import tqdm.rich as tqdm
from datetime import datetime
import os

P_CROSS = 0.5
GRAPH_SHAPE= (16, 16)
MAX_MUTATIONS = 5 # max number of mutations per making new individual
GRAPH_SIZE = GRAPH_SHAPE[0] * GRAPH_SHAPE[1]
P_DEVIATION = 0.5 #
TARGET_VALUE = 0.8*GRAPH_SIZE


today = datetime.now()

base_path = "./experiments/" + today.strftime('%Y%m%d%h%s') + 'lam_' + str(TARGET_VALUE)

class Individual:
    def __init__(self,):
        self.graph = None
        self.obstacles = []
        self.fitness = 100
        self.features = (0, 0)
        self.lam2_value = 100

class Container:
    def __init__(self, shape=(16, 1000), feature_domain=((0, .6), (0, 1))):
        self.grid = {(i, j): None for i in range(shape[0]) for j in range(shape[1])}
        self.shape = shape
        self.feature_domain = feature_domain
        self.individuals = []
        self.n_updated_bins = 0

    def insert(self, ind):
        features = ind.features
        to_bin = [round((features[0] / self.feature_domain[0][1]) * (self.shape[0] - 1)), round((features[1] / self.feature_domain[1][1]) * (self.shape[1] - 1))]
        to_bin[0] = min(to_bin[0], self.shape[0] - 1)
        to_bin[1] = min(to_bin[1], self.shape[1] - 1)
        to_bin = tuple(to_bin)
        if self.grid[to_bin]:
            if ind.fitness < self.grid[to_bin].fitness: # If better (closer to target) update bin
                self.individuals.remove(self.grid[to_bin])
                self.grid[to_bin] = ind
                self.individuals.append(ind)
                self.n_updated_bins += 1
        else:
            print("new cell!")
            self.grid[to_bin] = ind
            self.individuals.append(ind)

    def size(self):
        return len(self.individuals)

def percent_obstacles(ind):
    return len(ind.obstacles) / GRAPH_SIZE

def diameter(ind):
    return nx.diameter(ind.graph)

def is_adjacent(p1, p2):
    return (abs(p1[0]-p2[0]) == 1 and p1[1] == p2[1]) or (p1[0] == p2[0] and abs(p1[1] - p2[1]) == 1)

def is_valid(ind):
    g = ind.graph
    components = nx.connected_components(g)
    return len(list(components)) == 1 and g.number_of_nodes() > GRAPH_SIZE / 2


def obstacle_density(ind):
    if len(ind.obstacles) == 0:
        return 0
    obstacles = ind.obstacles
    adjacent_obstacles = 0
    for i in range(len(obstacles)):
        for j in range(i, len(obstacles)):
            o1 = obstacles[i]
            o2 = obstacles[j]

            if is_adjacent(o1, o2):
                adjacent_obstacles += 1
    return adjacent_obstacles / len(obstacles)

def lam2_val(ind):
    return ind.lam2_value

# Features in container
# If you modify features, you have to modify feature domain in the container
features1 = diameter
features2 = lam2_val
features1_range = [32, 16]
features2_range = [1e-6, 1e-2]

def eval_fn_l2(ind):
    graph = ind.graph
    lambda2 = nx.normalized_laplacian_spectrum(graph)[1]
    ind.fitness = (lambda2 - TARGET_VALUE)**2
    ind.lam2_value = lambda2
    ind.features = (features1(ind), features2(ind))
    return ind.fitness, ind.features

def eval_fn_n_obs(ind):
    graph = ind.graph
    lambda2 = nx.normalized_laplacian_spectrum(graph)[1]
    ind.lam2_value = lambda2
    ind.fitness = (ind.graph.number_of_nodes() - TARGET_VALUE)**2
    ind.features = (features1(ind), features2(ind))

    return ind.fitness, ind.features 

eval_fn = eval_fn_n_obs

def emit(container):
    num_repeats = np.random.randint(1, MAX_MUTATIONS)
    for _ in range(num_repeats):
        if np.random.random() < P_CROSS:
            individuals = np.random.choice(container.individuals, 2)
            new_ind = cross(individuals[0], individuals[1])
        else:
            ind = np.random.choice(container.individuals)
            new_ind = random_mutate(ind)
        if(not is_valid(new_ind)):
            return None
    return new_ind

def random_mutate(ind):
    new_ind = copy.deepcopy(ind)
    # if new_ind.lam2_value < TARGET_VALUE or np.random.random() < P_DEVIATION:
    if np.random.random() < P_DEVIATION:
        # Need to increase connectivity, add back in an obstacle
        try:
            obstacle_to_remove = random.choice(new_ind.obstacles)
        except:
            return new_ind 
        new_ind.obstacles.remove(obstacle_to_remove)
        new_ind.graph.add_node(obstacle_to_remove)
        x, y = obstacle_to_remove[0], obstacle_to_remove[1]
        if (x, y+1) not in new_ind.obstacles:
            new_ind.graph.add_edge((x, y+1), (x, y))
    else:
        to_remove = random.choice(list(new_ind.graph.nodes))
        new_ind.obstacles.append(to_remove)
        new_ind.graph.remove_node(to_remove)
    return new_ind

def cross(ind1, ind2):
    # TODO: Cross two individuals, randomly select vertical/horizontal line and split along that line
    new_ind = Individual()
    all_obstacles = list(set(ind1.obstacles).union(set(ind2.obstacles)))
    if len(all_obstacles) == 0:
        return copy.deepcopy(ind1)
    n_obstacles = random.randint(len(all_obstacles) // 2, len(all_obstacles))
    new_ind.obstacles = [all_obstacles[i] for i in np.random.choice(len(all_obstacles), n_obstacles, replace=False)]
    new_ind.graph = nx.grid_2d_graph(*GRAPH_SHAPE)
    for o in new_ind.obstacles:
        new_ind.graph.remove_node(o)
    return new_ind

def save_graph(filename, ind):
    pos = {(x, y): (x, y) for x, y in ind.graph.nodes()}
    plt.clf()
    nx.draw(ind.graph, pos)
    plt.savefig(filename)
    plt.clf()

def make_graphs(container, save=True, iter=None, display_graphs=False, save_state=False):
    fitness = []
    for i in range(container.shape[0]):
        fitness.append([])
        for j in range(container.shape[1]):
            if container.grid[(i, j)]:
                fitness[-1].append(container.grid[(i, j)].fitness)
            else:
                fitness[-1].append(0.1)
    # cmap = matplotlib.colormaps.get_cmap('PuOr')
    # cmap.set_bad(color='red')
    plt.imshow(fitness, aspect='auto')
    if save:
        if iter is not None:
            plt.savefig(base_path + f'iteration_{iter}_size_{container.size()}.pdf')
        else:
            plt.savefig(base_path + 'container.pdf')
    else:
        plt.show()
    
    if display_graphs:
        if not os.path.exists(base_path + f'graphs/iters_{iter}'):
            os.makedirs(base_path + f'graphs/iters_{iter}')
        for (k1, k2), v in container.grid.items():
            if v is not None:
                save_graph(base_path + f'graphs/iters_{iter}/{k1}_{k2}_{v.fitness}.pdf', v)
    
    if save_state:
        with open(base_path + f'iteration_{iter}_container.pkl', 'wb') as f:
            pickle.dump(container, f)

def run_experiment(budget=10000, target_value=None):
    global TARGET_VALUE # if target_value is not given, grab a constant target L2
    global base_path # Base path for experiments

    if target_value:
        TARGET_VALUE = target_value

    base_path = "./experiments/" + today.strftime('%Y%m%d%h%s') + 'lam_' + str(TARGET_VALUE)
    os.mkdir(base_path)
    base_path += '/'
    container_sizes = []
    updated_bins = []

    # Make initial graph (2d-grid graph), find L2 value and insert into container
    initial_graph = nx.grid_2d_graph(*GRAPH_SHAPE)
    ind = Individual()
    ind.graph = initial_graph
    eval_fn(ind)
    container = Container(feature_domain=(features1_range, features2_range))
    container.insert(ind)

    for i in tqdm.tqdm(range(budget)):
        if i % 10000 == 0 and i != 0:
            # Update experiment tracking, make graphs, save container (with all graphs)
            make_graphs(container, iter=i, save=True, display_graphs=False, save_state=True)
            container_sizes.append(container.size())
            updated_bins.append(container.n_updated_bins)
            print(updated_bins)
        new_ind = emit(container) # Make new individual from existing individuals
        if not new_ind:
            # print("not valid...")
            continue
        fitness, features = eval_fn(new_ind) # eval makes some in place modifications as well, such as setting new_ind.lam2_value
        container.insert(new_ind)
 
    print(container.size())
    plt.savefig(base_path + 'container_sizes.pdf')
    plt.plot(container_sizes)
    plt.plot(updated_bins) 
    plt.savefig(base_path + 'updated_bins.pdf')
    make_graphs(container)

if __name__ == '__main__':
    # for target_l2 in np.arange(1e-3, 2e-3, 1e-4):
        # run_experiment(100000, target_l2)
    run_experiment(1000000)