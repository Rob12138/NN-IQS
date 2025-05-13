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

## Figure 5
Figure 5 is two box plots to show the relative error distribution.
### dataset
There are several important points to note about the dataset.

-The dataset contains relative error for all up-scaling ratios. That is, seen ratios x2, x3, x4, and unseen ratios x6, x8, x10.

-For each ratio, there are 8 corresponding files.

For example, for x2, you would have 2_times, 2_times interpo, 2_times_cubic, 2_times_bicubic and each of their transition versions.

They each stand for NN-IQS prediction, bilinear interpolation, separate cubic interpolation, bicubic interpolation, and their corresponding statistics within the transition region.

-For the whole phase diagram files, relative error is stored in a 2D array form.

Each point on the phase diagram is error estimated separately.

For the transition region files, the wanted data points are further picked from the array to form a 1D list.

The difference can also be seen from the import in the ipynb file.
### ipynb file
This file import all the data and plot the graph.

Note that the import steps may have a little difference for the two types of files, and that the directory may need to be adjusted.