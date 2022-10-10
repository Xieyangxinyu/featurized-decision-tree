# Code
## Abstract
We provide documented code to (1) reproduce the main results in Section 4 of the manuscript and (2) reproduce the simulation results in Appendix H, with sufficient computing time. The code should allow users to become familiar with the required inputs of the models and how the data should be structured.

## Description
- How delivered: The main proposed methods are written in python3 and the rest methods are written in R 4.1.0. 

- Hardware requirements: The simulation study is conducted under [Harvard FAS Research Computing](https://www.rc.fas.harvard.edu/about/cluster-architecture/).

## Instructions for Use
- What is to be reproduced: Figure 1 in Section 4 and Figure 6-15 in Appendix H. 
- How to reproduce analyses:
  1. On the odyssey cluster, go to `featurized-decision-tree/python/experiments/expr/work/` to run `vs_exec1.sh`-`vs_exec7.sh`. 
  2. Go to `featurized-decision-tree/R/` to run `command_exec1.sh`-`command_exec7.sh`. 
  3. After all jobs are completed. Run `featurized-decision-tree/visual.R` to generate all figures. 
  4. Expected run-time of the workflow: Approximately 20 mins in python and 30 mins in R for a single simulated dataset.
  
