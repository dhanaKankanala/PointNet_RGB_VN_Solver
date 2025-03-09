import os
import random

import networkx as nx
import numpy as np
import pandas as pd
from pyntcloud import PyntCloud

horizontal = 1
# Vertical indicates b parameter in ellipse layout.
# Circular layout is a special case of ellipse layout when vertical=1.
vertical = 1
# resolution indicates resolution parameter in spiral layout.
resolution = 0.35
# coloring = ['gray', 'random', 'uniform']
coloring = 'uniform'
# visualization = ['ellipse', 'random', 'spiral']
visualization = 'circular'
# model_size = ['small', 'medium']
model_size = 'medium'
# by default node_size = 0.5 and edge_size=0.1
node_size = 0.5
edge_width = 0.1

random.seed(23)
np.random.seed(23)
min_node = max_node = 1
if model_size == 'medium':
    min_node = 20
    max_node = 50
if model_size == 'small':
    min_node = 4
    max_node = 20


def circular_layout_3d(G, dim, radius):
    pos_2d = nx.circular_layout(G, dim=dim - 1)
    # radius
    num_nodes = len(G.nodes())
    theta = np.linspace(0, 2 * np.pi, num_nodes)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    # theta = np.linspace(0, 2 * np.pi, len(edges) + 1)[:-1]
    # node_positions = {node: (np.cos(angle), np.sin(angle)) for node, angle in zip(range(1, len(edges) + 1), theta)}

    # Assign z-coordinates based on some metric (you need to replace this with your logic)
    # z_coordinates = {node: np.random.uniform(0, 1) for node in num_nodes}

    # based on position of node
    # z_coordinates = {nd: pos[1] for nd, pos in pos_2d.items()}

    # Calculate center of the graph
    # center = np.mean(list(pos_2d.values()), axis=0)
    # Assign z-coordinate based on the Euclidean distance from the center
    # z_coordinates = {nd: np.linalg.norm(np.array(pos_2d[nd]) - center) for nd in G.nodes()}

    # Calculate degree centrality for each node
    degree_centrality = nx.degree_centrality(G)
    # Assign Z-coordinates based on degree centrality
    z_coordinates = {nd: degree_centrality[nd] for nd in G.nodes}

    # Combine x, y, and z coordinates
    # pos_3d = {nd: (*pos_2d[nd], z_coordinates[nd]) for nd in G.nodes()}
    pos_3d = {nd: (x[i], y[i], z_coordinates[i]) for i, nd in enumerate(G.nodes)}
    return pos_3d


def graph_features(G, node):
    # graph Topology
    degree_centarlity = nx.degree_centrality(G)  # degree centrality of the graph
    clustering_coefficient = nx.clustering(G)  # clustering coeff of graph
    density = nx.density(G)  # graph density -no.of edges/max possible edges value
    # Graph Connectivity
    num_connected_components = nx.number_connected_components(G)
    # local features
    local_features = {node: list(nx.neighbors(G, node)) for node in G.nodes()}
    # Calculate the mean of local features for each node
    mean_local_features = {node: np.mean(features) for node, features in local_features.items()}
    # graph size
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    return degree_centarlity[node],density, num_connected_components, mean_local_features[node]


visualization = coloring + '_color_' + visualization

if 'ellipse' in visualization:
    visualization = str(vertical).replace('.', 'p') + '_' + visualization

if 'spiral' in visualization:
    visualization = str(resolution).replace('.', 'p') + '_sp_' + visualization

if node_size != 0.5:
    visualization = 'node_size_' + str(node_size).replace('.', 'p') + '_' + visualization

if edge_width != 0.1:
    visualization = 'edge_width_' + str(edge_width).replace('.', 'p') + '_' + visualization

print(visualization)
i = -1
cycles = ['hamiltonian', 'non_hamiltonian']
for each in cycles:
    CSV_DIR = 'C:/Users/dhana/Downloads/VN-Solver-main/VN-Solver-main/RGB_VN-Solver/CSV_files'
    os.makedirs(CSV_DIR, exist_ok=True)
    ADJ_DIR = 'C:/Users/dhana/Downloads/VN-Solver-main/VN-Solver-main/RGB_VN_Solver_Files/ADJ_files'
    os.makedirs(ADJ_DIR, exist_ok=True)
    # PLCE_DIR = 'C:/Users/dhana/Downloads/VN-Solver-main/VN-Solver-main/RGB_VN_Solver_Files/PLCE_files'
    # os.makedirs(PLCE_DIR, exist_ok=True)
    graph_files = [file for file in os.listdir(each) if file.endswith('.mat')]
    g = 0
    for file in graph_files:
        flag = True
        with open(str(each) + '/' + str(file), 'r') as f:
            matrices_text = f.read().strip().split('\n\n')

        graphs = []
        matrixs = []
        for matrix_text in matrices_text:
            if '1' in matrix_text:
                count = matrix_text.count('\n')
                if count >= min_node and count < max_node:
                    lines = matrix_text.strip().split('\n')
                    matrix = np.loadtxt(lines, dtype=int)
                    matrixs.append(matrix)
                    graph = nx.from_numpy_array(matrix)
                    graphs.append(graph)
        g = len(graphs) + g
        # print('len(graphs)',len(graphs))
        for j in range(len(graphs)):
            i += 1
            # Convert the 2D graph nodes into 3D coordinates (x, y, z)
            if visualization.endswith('random'):
                coordinates = nx.random_layout(graphs[j], dim=3)  # Layout algorithm
            if 'circular' in visualization:
                # coordinates = nx.spring_layout(graphs[j], dim=3)
                coordinates = nx.circular_layout(graphs[j], dim=3)  # Layout algorithm
                # coordinates = circular_layout_3d(graphs[j], dim=3, radius=10)
                # print(position_3d,"pos")
            if 'spiral' in visualization:
                coordinates = nx.spiral_layout(graphs[j], resolution=resolution, dim=3)  # Layout algorithm
            if 'ellipse' in visualization:
                coordinates = nx.ellipse_layout(graphs[j], horizontal=horizontal, vertical=vertical, dim=3)
            # Extract edge relationships
            all_edges = graphs[j].edges()

            # Create a dictionary to store edge relationships
            edge_relationships = {}

            # Iterate through all edges and build the edge relationships
            for source, target in all_edges:
                if source not in edge_relationships:
                    edge_relationships[source] = []
                edge_relationships[source].append(target)

            # Save the 3D coordinates as a CSV file
            csv_filename = os.path.join(CSV_DIR, f'graph_{each}_{j}_3d.csv')
            point_nodes = []
            Node_edge_points = []
            with open(csv_filename, 'w') as csvfile:
                csvfile.write('x,y,z,source,target\n')  # Write CSV header
                for node, (x, y, z) in coordinates.items():
                    type_param = 1
                    Node_edge_points.append(type_param)
                    if node in edge_relationships:
                        src_coordinates = np.array(coordinates[node])
                        targets_list = np.array(edge_relationships[node])
                        features = graph_features(graphs[j], node)
                        # node_point = np.array(np.append(coordinates[node], [type_param, node, node]))
                        np_features = np.array(features)
                        node_point1 = np.array(
                            np.append(coordinates[node], [type_param, node, node, len(targets_list),0]))
                        node_point = np.append(node_point1, np_features)
                        point_nodes.append(node_point)
                        tgt_coord_list = []
                        for k in range(len(targets_list)):
                            tgt_coordinates = np.array(coordinates[targets_list[k]])
                            tgt_coord_list.append(tgt_coordinates)
                            edge_len = edge_length = np.linalg.norm(tgt_coordinates - src_coordinates)
                            # distance = np.linalg.norm(tgt_coordinates - src_coordinates)
                            # Adjust the number of points based on the distance
                            # num_points = max(2, int(distance / 0.04))  # Adjust 0.1 as needed
                            num_points = 15
                            for pt in range(num_points):
                                type_param = 0
                                points = np.append(
                                    [src_coordinates + (tgt_coordinates - src_coordinates) * (pt / num_points)],
                                    [type_param, node, targets_list[k], 0, edge_len])
                                edge_point = np.append(points, np_features)
                                point_nodes.append(edge_point)
                                Node_edge_points.append(type_param)

                        targets = ','.join(map(str, edge_relationships[node]))
                        csvfile.write(
                            f'{x},{y},{z},{node},"{targets}"\n')  # Write node coordinates and edge relationships
                    else:
                        node_point1 = np.array(np.append(coordinates[node], [type_param, node, node, 0, 0]))
                        # node_point = np.array(np.append(coordinates[node], [type_param, node, node]))
                        node_point = np.append(node_point1, np_features)
                        point_nodes.append(node_point)
                        csvfile.write(f'{x},{y},{z},{node}, \n')  # Write node coordinates without edge relationships
                graph_point_cloud_data = point_nodes
            # Create a PyntCloud object and add points
            cloud = PyntCloud(pd.DataFrame(graph_point_cloud_data,
                                           columns=['x', 'y', 'z', 'Node', 'Source', 'Target', 'Degree', 'EdgeInfo',
                                                    'Dcentrality','Density', 'Connectivity',
                                                    'LocalFeat']))

            # cloud = PyntCloud(pd.DataFrame(graph_point_cloud_data, columns=['x', 'y', 'z', 'Node', 'Source', 'Target', 'Degree', 'EdgeInfo', 'Dcentrality', 'ClusteringCoef', 'Gdensity', 'Connectivity', 'LocalFeat', 'LNodes', 'LEdges']))
            # Save the point cloud to a file
            os.makedirs('Point_cloud_12features_15_' + visualization + '_' + str(each) + '_' + model_size, exist_ok=True)
            cloud.to_file(
                'Point_cloud_12features_15_' + visualization + '_' + str(each) + '_' + model_size + '/' + str(each) + '_' + str(
                    i) + '.ply')

            # save the adjacency matrix as needed
            adjacency_matrix = nx.to_numpy_array(graphs[j], dtype=int)
            Adj_mat_file = os.path.join(ADJ_DIR, f'adjacency_matrix_{each}_{j}.npy')
            np.save(Adj_mat_file, adjacency_matrix)
    print("total graphs", g, each, i)
    # csv_folder = 'C:/Users/dhana/Downloads/VN-Solver-main/VN-Solver-main/RGB_VN_Solver_Files/CSV_files'

    # Get a list of CSV files in the folder
    # csv_files = [file for file in os.listdir(csv_folder) if file.endswith('.csv')]

print(i)
