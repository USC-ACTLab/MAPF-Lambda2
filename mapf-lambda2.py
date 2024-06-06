# This is a simple example of how to calculate the lambda2 of map using networkx library
import networkx as nx
import numpy as np
import yaml

map_folder = "./dataset/ICAPS-24/"
map_name = "random-32-32-10.yaml"
yaml_map = map_folder + map_name

# load map
with open(yaml_map) as f:
    map_data = yaml.load(f, Loader=yaml.FullLoader)
dim_x = map_data["dimensions"][0]
dim_y = map_data["dimensions"][1]
obs = map_data["obstacles"]

# ceate graph
G = nx.Graph()
movements = [(-1, 0), (0, 1), (1, 0), (0, -1)]

# add nodes
for i in range(dim_x):
    for j in range(dim_y):
        if [i, j] not in obs:
            G.add_node((i, j))

# add edges
for i in range(dim_x):
    for j in range(dim_y):
        if [i, j] not in obs:
            # check neighbors
            for mi, mj in movements:
                if -1 < mi + i < dim_x and -1 < mj + j < dim_y and [mi + i, mj + j] not in obs:
                    G.add_edge((i, j), (mi + i, mj + j))

# calculate eigen values
eigen = nx.normalized_laplacian_spectrum(G)
print('lambda2 is: %f' % eigen[1])
