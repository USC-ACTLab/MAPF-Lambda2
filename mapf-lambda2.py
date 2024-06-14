# This is a simple example of how to calculate the lambda2 of map using networkx library
import networkx as nx
import numpy as np
import yaml

map_folder = "./dataset/MAPF_Benchmark/"
map_name = "maze-32-32-4.map"
map_path = map_folder + map_name

# load yaml map
if map_path.split('.')[-1] == "yaml":
    with open(map_path) as f:
        map_data = yaml.load(f, Loader=yaml.FullLoader)
    dim_x = map_data["dimensions"][0]
    dim_y = map_data["dimensions"][1]
    obs = map_data["obstacles"]
# load MAPF benchmark map
elif map_path.split('.')[-1] == "map":
    with open(map_path) as f:
        if f.readline() != "type octile\n":
            print("Bad mapfile!")
            exit()
        else:
            dim_y = int(f.readline().split(" ")[-1])
            dim_x = int(f.readline().split(" ")[-1])
            f.readline()
            x = y = 0
            obs = []
            for line in f.readlines():
                for l in line:
                    if l != ".":
                        obs.append([x, y])
                    x += 1
                x = 0
                y += 1
else:
    print("Only support .yaml and .map format!")
    exit()

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
print('map: %s' % map_name)
print('lambda2 is: %f' % eigen[1])

