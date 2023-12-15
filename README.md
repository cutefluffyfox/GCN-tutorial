# Introduction to graph neural networks

This is GitHub repository that is aimed to help newcomers into Graph NN to understand
basics of graph convolution method, how classification on graphs work and what are the 
common practices in this sphere.

This tutorial contains information how to implement graph convolution on raw numpy, but
it is worth mentioning that this tutorial can be extended (it already has all code, 
we made an [open-source pytorch-like library](https://pypi.org/project/inno-graph/) 
that is written on numpy) to even be used as introduction to linear models and how 
forward/backward passes works.


# How to run

The only file you need to run is Tutorial.ipynb. But before running please make sure
you have `networkx`, `matplotlib` and `numpy` installed. If no, you can 
run next command bellow:

```shell
pip install -r requirement.txt
```

Worth mentioning that you may require `inno-graph` library if you will use
tutorial independently (outside this repository).

To run notebook in jupyter-notebook environment you can use any IDE, service, or even
from terminal with `jupyter-notebook` command.


# Some insides into tutorial

We used [Karate-club](https://en.wikipedia.org/wiki/Zachary%27s_karate_club) dataset, 
however it was not our first iteration. From the beginning
we decided to use [IMDB-BINARY](https://paperswithcode.com/dataset/imdb-binary) 
dataset. The main problem we faced is that it's not that simple as we thought it was,
it was hard to understand (in [deprecated version](https://github.com/cutefluffyfox/GCN-tutorial/blob/15a03c13643968e996db28f5ea5c0271195228e6/notebooks/01-Learning-dataset.ipynb) 
we dedicated notebook to fully-explain it). We also were sceptic that it was simple to train
and our [Pytorch-implementation](https://github.com/cutefluffyfox/GCN-tutorial/blob/15a03c13643968e996db28f5ea5c0271195228e6/notebooks/02-Training-PyTorch-GNN.ipynb) 
proved that. For such reasons we decided to switch to Karate-club.

In addition, we wanted to provide in-depth tutorial with examples, so we decided to
implement full GCN-network by only using numpy. All interested people would have a chance
to implement all layers by hand, or import some from our library. Flexibility - our 
strong part!

# Repository structure
```
GCN-tutorial
│   README.md       # Introduction to project (this file)
│   Tutorial.ipynb  # Main tutorial file we worked on
│
└───igraph          # Pytorch-styled library we developed
│   │   layers
│   │   losses
│   │   models
│   │   preprocessing
│   
└───examples        # Examples how to use our library
    │   train_test_example.py
    

Other files - legacy / important for other reasons
```

# Credits 
This tutorial was created by:
* Polina Zelenskaya - responsible for library design, layers and tutorial
* Lev Rekhlov - responsible for library testing, models and tutorial
* Said Kamalov - responsible for library losses, preprocessing and tutorial

