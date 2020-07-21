# ComE_BGMM
Implementation of the paper 
**Learning Community Embedding with Community Detection and Node Embedding on Graphs**

With alteration by Anton Begehr: using bayesian gaussian mixture models instead of standard gaussian mixture models.

The implementation is base on Python 3.6.1 and Cython 0.25

The core algorithm is written in Cython, so we provide a miniconda environment file to run our code. 

## Conda Environment BICE

To create the BICE conda environment from env.yml, run `conda env create -f env.yml` and activate with `conda activate BICE`.

More details on conda environments here: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file

## GMM => BGMM progress:

Using `sklearn.mixture.BayesianGaussianMixture` for community embeddings.

## next up

Did `conda update --all` =>
- [x] save new env.yml
- [x] get main.py to run
- [x] fix graph_utils due to networkx update
- [x] run on karate club data (get directly from networkx example data)
