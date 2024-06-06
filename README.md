# MAPF-Lambda2
This is the code repo for ICAPS-24 paper: "[Map Connectivity and Empirical Hardness of Grid-based Multi-Agent Pathfinding Problem](http://idm-lab.org/bib/abstracts/papers/icaps24c.pdf)"

# Dataset

The following figure shows the detailed map information for Experiment 1 in the paper. $\lambda_2$ is annotate on x-axis and the smallest $\lambda_2$ map is at the top left corner. Maps from the [MovingAI benchmark](https://movingai.com/benchmarks/mapf/index.html) dataset are marked in red. All other maps are generated using our fractal and QD instance generator. The map files can be found in `dataset/ICAPS-24`.

![map info](pics/icaps24_exp_1.png)

# Usage

You can find the QD map generator in `qd-generator/`.

The `mapf-lambda2.py` is a sample script to calculate the $\lambda_2$ of a map.

More updates will come very soon including several MAPF map generators and a well documented code base.

Please cite the following paper if you find the results in this paper is helpful:

[1.] Ren, J., Ewing, E., Kumar, T. S., Koenig, S., & Ayanian, N. (2024). Map Connectivity and Empirical Hardness of Grid-based Multi-Agent Pathfinding Problem. In 34th International Conference on Automated Planning and Scheduling.

