# Deep-Meta-Learning
This is the second assignment of the course *Automated Machine Learning*

In this assignment we used the N-way k-shot image classification setting on the *Omniglot* dataset. This was done by implementing the Model Agnostic Meta-Learning (MAML) algorithm.

## File Organization
*maml.py* contains the implementation of the MAML algorithm.

*dataloaders.py*  contains the data loader code.

*networks.py* contains the two backbone networks: a feed-forward classifier and a convolutional one. In the experiments, we use the convolutional backbone. 

*main.py* main.py: is the main script to run for experiments.