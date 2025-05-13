# NN-IQS
This is a private share of datasets

## Figure 1
Figure 1 is a show of general workflow of this project.
### ppt file
The quantum simulation graph and the neural network graph are manually plotted, to re-scale in the second last page, I have snapshotted the neural network to insert as an image.

The final page is also a screenshot.
### ipynb file
The pyplotted graphs are done with the ipynb file provided, available in jupyter notebook.

To show system size difference, I adjusted the colormap. To show parameter difference, I adjusted this demonstration z function.

```python 
Z = np.cos(X)*np.sin(Y)
```
## Figure 2 & 3
Figure 2 is a demonstration phase diagram. 

While Figure 3 is a theoretical calculation cited from literature with mu set to zero. 
Therefore the Figure 3 contour is a straight line and do not vary with mu.
### py file
The py file include all the functions needed in the ipynb file. It is constructed from 
the spin model provided in the paper.
### ipynb file
This file uses the written functions to plot demonstration graphs, including the simulated one and the theoretical one.

Something to notice here is the parameter choice.

I have forgotten the original parameter setting to plot the existing two graphs in the paper. However, in our further demonstration in the paper, we have always selected this parameter setting.

```python 
g = 3
N= 8
w = 3
m = 0
```
This is also consistent with our parameter range : w/g within [0.3, 1.5].

## Figure 4
Figure 4 is to show the training curve of the neural network.

It includes training loss, test loss, and test psnr.
### csv file
The csv file recorded the three values associated with each epoch.
### ipynb file
The ipynb file simply serves as a graph plot. The file directory might need to be adjusted.

