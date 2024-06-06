import yaml
import numpy as np
import matplotlib.pyplot as plt
class MAPF_Instance:

    def __init__(self, dimensions, obstacles=[], starts=[], goals=[]):
        assert(len(dimensions) == 2)
        self.dim = dimensions
        self.obs = obstacles
        self.starts = starts
        self.goals = goals
        self.map_np = np.zeros(self.dim)
        for o in self.obs:
            self.map_np[tuple(o)] = 1

    def add_obstacle(self, obs):
        self.obs.append(obs)
        self.map_np[obs] = 1


    def to_yaml(self, filename):
        agents_dict = [{'start': self.starts[i], 'name': 'agent{}'.format(i),
            'goal': self.goals[i]} for i in range(len(self.starts))]
        map_dict = {'dimensions': self.dim, 'obstacles': self.obs}

        instance_dict = {'agents': agents_dict, 'map': map_dict}
        with open(filename, 'w') as f:
            yaml.dump(instance_dict, f, default_flow_style=None)

    def agents_only_yaml(self, filename, map_path):
        agents_dict = [{'start': list(self.starts[i]), 'name': 'agent{}'.format(i),
            'goal': list(self.goals[i])} for i in range(len(self.starts))]
        with open(filename, 'w') as f:
            yaml.dump({'agents': agents_dict, 'map_path': map_path}, f)

    def map_only_yaml(self, filename):
        obs = [list(o) for o in self.obs]
        map_dict = {'dimensions': list(self.dim), 'obstacles': obs}
        with open(filename, 'w') as f:
            yaml.dump(map_dict, f, default_flow_style=None)
   
    def free_cells(self):
        open_cells = []
        for i in range(self.dim[0]):
            for j in range(self.dim[1]):
                if self.map_np[i, j] == 0:
                    open_cells.append((i, j))
        return open_cells

    def neighbors(self, cell):
        x = cell[0]
        y = cell[1]

        neighbors = []

        if x + 1 < self.dim[0]:
            neighbors.append([x+1, y])
        if y + 1 < self.dim[1]:
            neighbors.append([x, y+1])
        if x - 1 >= 0:
            neighbors.append([x-1, y])
        if y - 1 >= 0:
            neighbors.append([x, y-1])

        return neighbors

    def _find_sccs(self):
        unvisited = set()
        visited = set()

        open_cells = self.free_cells()
        [unvisited.add(tuple(cell)) for cell in open_cells]
        sccs = []
        while(unvisited):
            cell = unvisited.pop()
            visited.add(cell)
            scc = []
            visiting = set()
            visiting.add(cell)
            while visiting:
                c = visiting.pop()
                scc.append(c)
                visited.add(c)
                for n in self.neighbors(c):
                    n = tuple(n)
                    if n in unvisited:
                        visiting.add(n)
                        unvisited.remove(n)
            sccs.append(scc)

        return sccs
                
    def make_single_scc(self):
        sccs = self._find_sccs()
        scc_sizes = [len(scc) for scc in sccs]
        biggest_scc = np.argmax(scc_sizes)
        for i, scc in enumerate(sccs):
            if i == biggest_scc:
                continue
            for cell in scc:
                self.add_obstacle(cell)
        return scc_sizes[biggest_scc]
    
    def visualize_map(self):
        plt.imshow(self.map_np.transpose(), cmap='Greys')
        plt.show()
   
    def from_np_map(m):
        dim = m.shape
        instance = MAPF_Instance(dim, [], [], [])
        for i in range(m.shape[0]):
           for j in range(m.shape[1]):
               if m[i, j] == 1:
                    instance.add_obstacle((i,j))
        return instance

    def scale_borders(self):
        free = self.free_cells()

        left = min([c[0] for c in free])
        right = max([c[0] for c in free])
        bottom = min([c[1] for c in free])
        top = max([c[1] for c in free])

        print(left, right, bottom, top)

        new_map = self.map_np[left:right+1, bottom:top+1] 
        #self.obs = [o for o in self.obs if not (o[0] < 0 or o[0] >= new_dim[0] or o[1] < 0 or o[1] >= new_dim[1])]

        #map_np = np.zeros(new_dim)
        #for o in self.obs:
        #    map_np[tuple(o)] = 1
        self.obs = []
        for i in range(len(new_map)):
            for j in range(len(new_map[i])):
                if new_map[i, j] == 1:
                    self.obs.append((i,j))
        self.map_np = new_map
        self.dim = self.map_np.shape 
    
    
    def post_process(self):
        #Gather largest SCC and update boarders
        self.make_single_scc()
        self.scale_borders()
    
    def visualize_instance(self):
        plt.imshow(self.map_np, alpha=0.5, cmap='Greys')
        
        agents_np = np.zeros_like(self.map_np)
        goals_np = np.zeros_like(self.map_np)

        for a in self.starts:
            agents_np[a] = 1
        for g in self.goals:
            goals_np[g] = 1

        plt.imshow(agents_np, alpha=0.5, cmap='Greens')
        plt.imshow(goals_np, alpha=0.5, cmap='Reds')
        plt.show()

def from_benchmarks_map_file(filename):
    print(filename)
    with open(filename, 'r') as f:
        lines = f.readlines()
    dim = (len(lines[4]), len(lines)-4)
    print(dim)
    obstacles = []
    for y, line in enumerate(lines[4:]):
        for x, c in enumerate(line):
            if c in "@TO":
                obstacles.append((x,y))
    print(len(obstacles))
    instance = MAPF_Instance(dim, obstacles)
    instance.post_process()
    return instance

def load_map():
    filename = 'data/maps/fractal_0.4_(200, 200).yaml'
    with open(filename, 'r') as f:
        y = yaml.load(f)
    dim = y['dimensions']
    obs = y['obstacles']
    instance = MAPF_Instance(dim, obstacles=obs)
    instance.visualize_map()
    instance.post_process()
    instance.visualize_map()

def test_scc():
    dimensions = [10, 10]
    obstacles = [[1,0], [0,1]]
    starts = [[1,2], [9,9]]
    goals = [[5, 5], [2, 2]]
    
    instance = MAPF_Instance(dimensions, obstacles, starts, goals)

    dimensions = [10, 10]
    obstacles = [[1,0], [0,1], [5, 5], [7,5], [7,6], [5,6], [6,4], [6,7]]
    starts = [[1,2], [9,9]]
    goals = [[5, 5], [2, 2]]
    
    instance = MAPF_Instance(dimensions, obstacles, starts, goals)

    instance.make_single_scc()

    instance.visualize_map()


def test_yaml():
    dimensions = [10, 10]
    obstacles = [[1,1], [4,5]]
    starts = [[1,2], [9,9]]
    goals = [[5, 5], [2, 2]]

    instance = MAPF_Instance(dimensions, obstacles, starts, goals)
    instance.to_yaml('test.yaml')
    instance.map_only_yaml('map.yaml')

if __name__ == '__main__':
    load_map()
    
