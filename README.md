**RGB VN-Solver using pointCloud Classification

Overview**

This repository contains scripts for processing, training, testing, and visualizing point cloud data for graph classification. The project primarily focuses on differentiating between Hamiltonian and non-Hamiltonian graphs using deep learning techniques.

**Repository Structure**

layout.py - Contains node positioning algorithms for graph drawing, extending NetworkX's layout functionalities.

generate_fig_12feat.py - Generates 3D visualizations of point cloud graphs using different layouts such as circular, spiral, and elliptical.

train_12feat.py - Implements training procedures for a deep learning model (PointNet) on 12-feature point cloud graph datasets.

test_on_best_12feat.py - Evaluates the trained model on a test dataset with 12-feature point cloud data.

**RGB VN-Solver**
In the VN-Solver RGB folder, we have the code for generating figures from adjacency matrixes in generate_figure.py where the uniform configuration is used (nodes are red, edges are black). We introduced ellipse layout in layout.py which should be replaced by the layout.py in networkx package. Primarily, the size of the node circle is 0.5 and the thickness of the edge segment is 0.1. Further, we set different values for node_size and edge_width. After generating figures, we train 5 ResNet models using 3, 7, 11, 13, 29 as seeds to split the data. The random seed of the model is 23 and the models are trained on 2 GPUs. As we have datasets with 3 different sizes, we defined a tv variable to indicate the size of the training set and validation set. The test_on_best.py evaluates the model with the highest F1 score on the test set while the test_on_epochs.py evaluate models saved in each epoch against the test set. Evaluation of each epoch is further used to draw Figure 3 in the manuscript. The code to generate this figure is available in draw_figures_in_manuscript.py. After evaluating the best model on the test set, we aggregate the results of the seeds to have the final result. This is done in aggregate_seed.py.


**Dependencies**

Ensure the following Python libraries are installed before running the scripts:torch, torchvision, numpy, pandas, networkx, plyfile, pyntcloud, Usage

**Data Preparation:** Ensure that the point cloud dataset is structured correctly in PointCloud_12features_15_... directories.
**Training:** Run train_11feat.py to train the PointNet model.
**Testing:** Execute test_on_best_12feat.py to evaluate model performance.
**Visualization:** Use generate_fig_11feat.py to generate and save visual representations of graphs.

**Author**
Dhana Lakshmi Kankanala



