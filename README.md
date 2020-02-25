# ComE_BGMM
Implementation of the paper 
**Learning Community Embedding with Community Detection and Node Embedding on Graphs**

With alteration by Anton Begehr: using bayesian gaussian mixture models instead of standard gaussian mixture models.

The implementation is base on Python 3.6.1 and Cython 0.25

The core algorithm is only written in Cython, so we provide a miniconda environment file to run our code. 

## GMM => BGMM progress:

Not started.

## next up

Did `conda update --all` =>
- save new env.yml
- get main.py to run
- fix graph_utils due to networkx update
- run on karate club data (get directly from networkx example data)