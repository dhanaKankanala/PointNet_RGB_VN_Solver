import os
import random
import re
import shutil

import numpy as np
import numpy.random
import pandas as pd
import plyfile
import torch
# from plyfile import PlyData
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from pyntcloud import PyntCloud
from sklearn.metrics import roc_auc_score, f1_score, recall_score
from torch.utils.data import Dataset

torch.backends.cudnn.deterministic = True


def sort_key(string):
    # Extract numeric portion of the string using regular expression
    numeric_part = re.search(r'\d+', string).group()
    # Convert the numeric part to integer for comparison
    return int(numeric_part)


# choose among [3, 7, 11, 13, 29]
seed = 3
# set seed for reproducibility
random.seed(seed)
# select among random, ellipse, random
visualization = 'circular'
# select between medium and small
model_size = 'small'
# select between random or uniform
coloring = 'uniform'
# select  among 1, 2, or 3 for the train/val splits of 80/20, 160/40, or 800/200
tv = 1
node_size = 0.5
edge_width = 0.1
vertical = 1
resolution = 0.35

if tv == 1:
    train_size = 80
    val_size = 20
if tv == 2:
    train_size = 160
    val_size = 40
if tv == 3:
    train_size = 800
    val_size = 200

test_size = 500
size = 224
resize = False
params = None

if 'ellipse' in visualization:
    if params != None:
        params = str(vertical).replace('.', 'p') + '_' + params
        visualization = str(vertical).replace('.', 'p') + '_' + visualization
    else:
        params = str(vertical).replace('.', 'p') + '_' + coloring + "_" + visualization + "_" + str(
            model_size) + "_" + str(seed) + '_' + str(
            train_size) + "_" + str(test_size) + "_" + str(val_size)
        visualization = str(vertical).replace('.', 'p') + '_' + coloring + '_color_' + visualization

if node_size != 0.5:
    if params != None:
        params = 'node_size_' + str(node_size).replace('.', 'p') + '_' + params
        visualization = 'node_size_' + str(node_size).replace('.', 'p') + '_' + visualization
    else:
        params = 'node_size_' + str(node_size).replace('.', 'p') + '_' + coloring + "_" + visualization + "_" + str(
            model_size) + "_" + str(seed) + '_' + str(
            train_size) + "_" + str(test_size) + "_" + str(val_size)
        visualization = 'node_size_' + str(node_size).replace('.', 'p') + '_' + coloring + '_color_' + visualization

if edge_width != 0.1:
    if node_size == 0.5 and params == None:
        params = 'edge_width_' + str(edge_width).replace('.', 'p') + '_' + coloring + "_" + visualization + "_" + str(
            model_size) + "_" + str(
            seed) + '_' + str(
            train_size) + "_" + str(test_size) + "_" + str(val_size)
        visualization = 'edge_width_' + str(edge_width).replace('.', 'p') + '_' + coloring + '_color_' + visualization
    else:
        params = 'edge_width_' + str(edge_width).replace('.', 'p') + '_' + params
        visualization = 'edge_width_' + str(edge_width).replace('.', 'p') + '_' + visualization

if visualization == 'spiral':

    params = str(resolution).replace('.', 'p') + '_sp_' + coloring + "_" + visualization + "_" + str(
        model_size) + "_" + str(
        seed) + '_' + str(
        train_size) + "_" + str(test_size) + "_" + str(val_size)
    visualization = str(resolution).replace('.', 'p') + '_sp_' + coloring + '_color_' + visualization

elif params == None:
    params = coloring + "_" + visualization + "_" + str(model_size) + "_" + str(seed) + '_' + str(
        train_size) + "_" + str(test_size) + "_" + str(val_size)
    visualization = coloring + '_color_' + visualization
# params = visualization+"_"+str(model_size)+"_"+str(seed)+'_'+str(train_size)+"_"+str(test_size)+"_"+str(val_size)

# Define pointcloud directories
plce_graphs_non_hamiltonian_dir = 'Point_cloud_12features_15_' + visualization + '_non_hamiltonian_' + str(
    model_size) + '/'
plce_graphs_hamiltonian_dir = 'Point_cloud_12features_15_' + visualization + '_hamiltonian_' + str(model_size) + '/'
random.seed(seed)
all_point_Clouds = os.listdir(plce_graphs_non_hamiltonian_dir) + os.listdir((plce_graphs_hamiltonian_dir))
all_point_Clouds = sorted(all_point_Clouds, key=sort_key)
random.shuffle(all_point_Clouds)
pcloud_graph_data = []
label = {}

for file in all_point_Clouds:
    if file.endswith(".ply"):
        if 'non_hamiltonian' in file:
            label[file] = 0
            class_label = "non_hamiltonian"
            ply_data = plyfile.PlyData.read(os.path.join(plce_graphs_non_hamiltonian_dir, file))
            cloud = PyntCloud.from_file(os.path.join(plce_graphs_non_hamiltonian_dir, file))
        else:
            label[file] = 1
            class_label = "hamiltonian"
            ply_data = plyfile.PlyData.read(os.path.join(plce_graphs_hamiltonian_dir, file))
            cloud = PyntCloud.from_file(os.path.join(plce_graphs_hamiltonian_dir, file))
        vertices = ply_data['vertex']
        graph_points_coord = [
            [vertex['x'], vertex['y'], vertex['z'], vertex['Node'], vertex['Source'], vertex['Target'],
             vertex['Degree'], vertex['EdgeInfo'], vertex['Dcentrality'],vertex['Density'],
             vertex['Connectivity'], vertex['LocalFeat']] for vertex in vertices]
        pcloud_graph_data.append((graph_points_coord, label[file]))

# Create train, validation, and test directories

os.makedirs('PointCloud_12features_15_' + visualization + '/data_' + str(params) + '/train/hamiltonian', exist_ok=True)
os.makedirs('PointCloud_12features_15_' + visualization + '/data_' + str(params) + '/val/hamiltonian', exist_ok=True)
os.makedirs('PointCloud_12features_15_' + visualization + '/data_' + str(params) + '/test/hamiltonian', exist_ok=True)

os.makedirs('PointCloud_12features_15_' + visualization + '/data_' + str(params) + '/train/non_hamiltonian',
            exist_ok=True)
os.makedirs('PointCloud_12features_15_' + visualization + '/data_' + str(params) + '/val/non_hamiltonian',
            exist_ok=True)
os.makedirs('PointCloud_12features_15_' + visualization + '/data_' + str(params) + '/test/non_hamiltonian',
            exist_ok=True)

if os.path.exists('data_' + str(params) + '_saved_models'):
    # Remove the directory and all its contents
    shutil.rmtree('data_' + str(params) + '_saved_models')

# Create the new directory
os.makedirs('PointCloud_12features_15_' + visualization + '/data_' + str(params) + '_saved_models', exist_ok=True)
random.shuffle(all_point_Clouds)
print('all', len(all_point_Clouds))

# Split the sampled point clouds into train, validation, and test sets
for i, file in enumerate(all_point_Clouds[:train_size]):
    try:
        shutil.copy(os.path.join(plce_graphs_non_hamiltonian_dir, file),
                    os.path.join(
                        'PointCloud_12features_15_' + visualization + '/data_' + str(params) + '/train/non_hamiltonian',
                        file))
    except:
        shutil.copy(os.path.join(plce_graphs_hamiltonian_dir, file),
                    os.path.join(
                        'PointCloud_12features_15_' + visualization + '/data_' + str(params) + '/train/hamiltonian',
                        file))
for i, file in enumerate(all_point_Clouds[train_size:train_size + val_size]):
    try:
        shutil.copy(os.path.join(plce_graphs_non_hamiltonian_dir, file),
                    os.path.join(
                        'PointCloud_12features_15_' + visualization + '/data_' + str(params) + '/val/non_hamiltonian',
                        file))
    except:
        shutil.copy(os.path.join(plce_graphs_hamiltonian_dir, file),
                    os.path.join(
                        'PointCloud_12features_15_' + visualization + '/data_' + str(params) + '/val/hamiltonian',
                        file))
for i, file in enumerate(all_point_Clouds[train_size + val_size:train_size + val_size + test_size]):
    try:
        shutil.copy(os.path.join(plce_graphs_non_hamiltonian_dir, file),
                    os.path.join(
                        'PointCloud_12features_15_' + visualization + '/data_' + str(params) + '/test/non_hamiltonian',
                        file))
    except:
        shutil.copy(os.path.join(plce_graphs_hamiltonian_dir, file),
                    os.path.join(
                        'PointCloud_12features_15_' + visualization + '/data_' + str(params) + '/test/hamiltonian',
                        file))

seed_model = 23
torch.manual_seed(seed_model)
torch.cuda.manual_seed(seed_model)
numpy.random.seed(seed_model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# POINTNET model
class TNet(nn.Module):
    def __init__(self, input_fea):
        super(TNet, self).__init__()
        self.conv1 = nn.Conv1d(input_fea, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        # self.bn3 = nn.BatchNorm1d(256)
        # self.conv4 = nn.Conv1d(256, 1024, 1)
        self.bn4 = nn.BatchNorm1d(1024)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 144)
        # self.fc4 = nn.Linear(128, 64)
        # self.fc5 = nn.Linear(64, 36)  # 4x4 matrix for transformation
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # num_feat = x.size(1)
        x = self.conv1(x)
        x = torch.relu(x)
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        # x = torch.relu(self.conv4(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        # x = torch.relu(self.fc3(x))
        # x = torch.relu(self.fc4(x))
        x = self.fc3(x)

        # Add an identity matrix to the transformation matrix
        # identity = torch.eye(3, dtype=x.dtype, device=x.device).view(1, 9).repeat(batch_size, 1)
        # x = x + identity

        x = x.view(-1, 12, 12)
        return x


# PointNet classification module
class PointNetCls(nn.Module):
    def __init__(self, num_classes=2):
        super(PointNetCls, self).__init__()
        self.input_transform = TNet(12)
        self.feature_transform = TNet(128)
        self.conv1 = nn.Conv1d(12, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        # self.bn3 = nn.BatchNorm1d(256)
        # self.conv4 = nn.Conv1d(256, 1024, 1)
        self.bn4 = nn.BatchNorm1d(1024)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        # self.fc4 = nn.Linear(128, 64)
        # self.fc5 = nn.Linear(64, num_classes)  # 4x4 matrix for transformation
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, x):
        k = x
        # Input transformation
        trans = self.input_transform(x)
        x = torch.bmm(trans, x)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        # x = torch.relu(self.conv3(x))

        # Feature transformation
        trans = self.feature_transform(x)
        x = torch.bmm(trans, k)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        # x = torch.relu(self.conv4(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        # x = torch.relu(self.fc3(x))
        # x = torch.relu(self.fc4(x))
        x = self.fc3(x)
        return x


# Initialize the PointNet model
PointnetModel = PointNetCls(num_classes=2)
PointnetModel = nn.DataParallel(PointnetModel)  # Utilize multiple GPUs
PointnetModel.to(device)
# Print the model architecture
print(PointnetModel)
# PointnetModel.summary()

if resize == True:
    # Set up data loaders
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])
if coloring == 'gray':
    transform = transforms.Compose([
        transforms.Grayscale(),
        # transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])
else:
    transform = transforms.Compose([
        # transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])

classes = ["hamiltonian", "non_hamiltonian"]


class PointCloudDataset(Dataset):
    def __init__(self, pcdata, labels, transform=None):
        self.data_dir = pcdata
        self.label = labels
        self.transform = transform
        self.pcloud_graph_data1 = self._cloud_data()

    def _cloud_data(self):
        pcloud_graph_data1 = []
        for each in classes:
            plce_graphs = os.path.join(self.data_dir, each)
            data_dir = os.listdir(plce_graphs)
            for point_cloud_file in data_dir:
                ply_data = plyfile.PlyData.read(os.path.join(plce_graphs, point_cloud_file))
                vertices = ply_data['vertex']
                graph_points = [
                    [vertex['x'], vertex['y'], vertex['z'], vertex['Node'], vertex['Source'], vertex['Target'],
                     vertex['Degree'], vertex['EdgeInfo'], vertex['Dcentrality'],vertex['Density'], vertex['Connectivity'], vertex['LocalFeat']] for vertex in vertices]
                pcloud_graph_data1.append((graph_points, self.label[point_cloud_file]))
        return pcloud_graph_data1
        # for each in classes:
        #     plce_graphs = os.path.join(data, each)
        #     data_dir = os.listdir(plce_graphs)
        #     for point_cloud_file in data_dir:
        #         ply_data = plyfile.PlyData.read(os.path.join(plce_graphs, point_cloud_file))
        #         vertices = ply_data['vertex']
        #         graph_points = [[vertex['x'], vertex['y'], vertex['z'], vertex['Node'], vertex['Source'], vertex['Target']] for vertex in vertices]
        #         pcloud_graph_data1.append((graph_points, label[point_cloud_file]))
        # self.file = graph_points
        # self.label = labels[point_cloud_file]

    def __len__(self):
        return len(self.pcloud_graph_data1)

    def __getitem__(self, idx):
        point_cloud = self.pcloud_graph_data1[idx][0]
        label1 = self.pcloud_graph_data1[idx][1]
        if self.transform:
            augumented_point_cloud = augment(point_cloud)
            # Convert point cloud to PyTorch tensor
            # tensor_point_cloud = torch.tensor(augumented_point_cloud, dtype=torch.float32)

            # Apply the custom transformation
            transformed_point_cloud = self.transform(augumented_point_cloud)

            # Convert the transformed tensor back to a point cloud
            transformed_point_cloud = transformed_point_cloud.numpy().tolist()
            return transformed_point_cloud, label1
        else:
            return point_cloud, label1


# Custom transformation class for rotation
class PointCloudRotation(object):
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, point_cloud):
        # Assuming point_cloud is a PyTorch tensor
        # Extract x, y, z coordinates
        x, y, z, node, source, target, degree, edgeinfo, dcentrality,density,  connectivity, localfeat = point_cloud[
                                                                                                                                          :,
                                                                                                                                          0], point_cloud[
                                                                                                                                              :,
                                                                                                                                              1], point_cloud[
                                                                                                                                                  :,
                                                                                                                                                  2], point_cloud[
                                                                                                                                                      :,
                                                                                                                                                      3], point_cloud[
                                                                                                                                                          :,
                                                                                                                                                          4], point_cloud[
                                                                                                                                                              :,
                                                                                                                                                              5], point_cloud[
                                                                                                                                                                  :,
                                                                                                                                                                  6], point_cloud[
                                                                                                                                                                      :,
                                                                                                                                                                      7], point_cloud[
                                                                                                                                                                          :,
                                                                                                                                                                          8], point_cloud[
                                                                                                                                                                              :,
                                                                                                                                                                              9], point_cloud[
                                                                                                                                                                                  :,
                                                                                                                                                                                  10], point_cloud[:, 11]

        # Convert angle to radians
        angle_rad = np.radians(self.angle)

        # Apply rotation around the Z-axis
        x_rotated = x * np.cos(angle_rad) - y * np.sin(angle_rad)
        y_rotated = x * np.sin(angle_rad) + y * np.cos(angle_rad)

        # Combine rotated coordinates
        rotated_point_cloud = torch.stack((x_rotated, y_rotated, z, node, source, target, degree, edgeinfo, dcentrality,density,
                                        connectivity, localfeat), dim=1)

        return rotated_point_cloud


def calculate_num_points(dataset):
    total_points = 0
    for i in range(len(dataset)):
        total_points += len(dataset[i][0])
    return int(total_points / len(dataset))


def custom_collate_fn1(batch, num_points=64):
    # Extract the point clouds and labels from the batch
    point_clouds, labels = zip(*batch)
    # Create a tensor to store the point clouds in the batch
    batch_size = len(batch)
    # num_features = len(point_clouds[0][0][0])  # Assuming all point clouds have the same number of features
    point_cloud_tensor = torch.zeros(batch_size, 12, num_points)
    # Iterate over the point clouds in the batch and make sure each has num_points
    for i in range(batch_size):
        # Select num_points randomly from the original point cloud
        original_points = point_clouds[i]
        if len(original_points) < num_points:
            selected_points = original_points + [[0.0] * 12] * (num_points - len(original_points))
        else:
            selected_points = point_clouds[i][:num_points]
        selected_points = np.array(selected_points).T

        # Fill the tensor with the selected points
        point_cloud_tensor[i, :len(selected_points), :] = torch.tensor(selected_points)
    # Convert the labels to a tensor
    label_tensor = torch.tensor(labels)

    # print("avg_points",total_points/len(batch))
    return point_cloud_tensor, label_tensor


def augment(points):
    points = torch.FloatTensor(points)
    # jitter points
    points += torch.FloatTensor(points.shape).uniform_(-0.005, 0.005)
    # shuffle points
    indices = torch.randperm(points.shape[0])
    points = points[indices]
    return points


rotation_transform = PointCloudRotation(angle=30.0)

data_transform_pc = transforms.Compose([
    PointCloudRotation(angle=30.0)  # pointcloud rotation
    # transforms.RandomResizedCrop(512),      # Random resized crop
    # transforms.RandomHorizontalFlip(),  # Random horizontal flip
    # transforms.ToTensor(),  # Convert to PyTorch tensor
])

PC_train_dataset = PointCloudDataset('PointCloud_12features_15_' + visualization + '/data_' + str(params) + '/train/',
                                     label,
                                     transform=data_transform_pc)
PC_val_dataset = PointCloudDataset('PointCloud_12features_15_' + visualization + '/data_' + str(params) + '/val/',
                                   label,
                                   transform=data_transform_pc)
PC_test_dataset = PointCloudDataset('PointCloud_12features_15_' + visualization + '/data_' + str(params) + '/test/',
                                    label,
                                    transform=data_transform_pc)
print("lengths", len(PC_test_dataset), "train", len(PC_train_dataset), "val", len(PC_val_dataset))
batch_size = 32
train_points = calculate_num_points(PC_train_dataset)
val_points = calculate_num_points(PC_val_dataset)
test_points = calculate_num_points(PC_test_dataset)
print("pts", test_points, val_points, test_points)
PC_train_loader = torch.utils.data.DataLoader(PC_train_dataset, batch_size=batch_size, shuffle=True,
                                              collate_fn=lambda batch: custom_collate_fn1(batch,
                                                                                          num_points=train_points))
PC_val_loader = torch.utils.data.DataLoader(PC_val_dataset, batch_size=batch_size, shuffle=False,
                                            collate_fn=lambda batch: custom_collate_fn1(batch, num_points=val_points))
PC_test_loader = torch.utils.data.DataLoader(PC_test_dataset, batch_size=batch_size, shuffle=False,
                                             collate_fn=lambda batch: custom_collate_fn1(batch, num_points=test_points))

# Define loss function
criterion = nn.CrossEntropyLoss()
weight_decay = 1e-4
# Define optimizer and learning rate scheduler
optimizer = optim.Adagrad(PointnetModel.parameters(), lr=0.001, weight_decay=weight_decay)
# scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.09)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
# scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.01, gamma=0.1)

# Training loop
num_epochs = 100  # Adjust as needed
best_val_f1 = 0.0
early_stopping_patience = 8
early_stopping_counter = 0
metrics_df = pd.DataFrame(columns=['Epoch', 'AUC', 'Accuracy', 'F1', 'Recall'])

print("Started Training")

# Training loop
for epoch in range(num_epochs):
    epoch = epoch + 1
    print(f"Epoch {epoch}/{num_epochs}")
    print('-' * 10)
    # Training phase
    PointnetModel.train()
    train_loss = 0.0
    train_correct = 0

    for i, data in enumerate(PC_train_loader):
        inputs, labels1 = data
        inputs, labels1 = inputs.to(device), labels1.to(device)
        optimizer.zero_grad()

        outputs = PointnetModel(inputs)
        loss = criterion(outputs, labels1)
        l2_regularization = 0.0
        for param in PointnetModel.parameters():
            l2_regularization += torch.norm(param, 2)

        loss += weight_decay * l2_regularization

        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        train_loss += loss.item() * len(inputs[1])
        train_correct += torch.sum(preds == labels1)
    train_loss = train_loss / len(PC_train_dataset)
    train_acc = train_correct.double() / len(PC_train_dataset)
    # Save the model after each epoch
    model_path = os.path.join('PointCloud_12features_15_' + visualization + '/data_' + str(params) + '_saved_models',
                              f'model_epoch_{epoch}.pt')
    torch.save(PointnetModel.state_dict(), model_path)

    # Validation phase
    PointnetModel.eval()
    val_loss = 0.0
    val_correct = 0

    y_true_list = []
    y_pred_list = []
    y_pred_prob_list = []

    with torch.no_grad():
        for i, data in enumerate(PC_val_loader):
            inputs, labels1 = data
            inputs, labels1 = inputs.to(device), labels1.to(device)

            outputs = PointnetModel(inputs)
            loss = criterion(outputs, labels1)
            l2_regularization = 0.0
            for param in PointnetModel.parameters():
                l2_regularization += torch.norm(param, 2)

            loss += weight_decay * l2_regularization

            _, preds = torch.max(outputs, 1)
            val_loss += loss.item() * len(inputs[1])
            val_correct += torch.sum(preds == labels1)

            labels_np = labels1.cpu().numpy()
            preds_np = preds.cpu().numpy()

            # Append true labels and predicted probabilities to the lists
            y_true_list.extend(labels_np.tolist())
            y_pred_list.extend(preds_np.tolist())
            y_pred_prob_list.extend(outputs[:, 1].cpu().detach().numpy().tolist())
        val_loss = val_loss / len(PC_val_dataset)
        val_acc = val_correct.double() / len(PC_val_dataset)
        val_acc = val_acc.cpu().numpy()
        # Calculate metrics
        y_true = np.array(y_true_list)
        y_pred = np.array(y_pred_list)
        y_pred_prob = np.array(y_pred_prob_list)
        auc = roc_auc_score(y_true, y_pred_prob)
        f1 = f1_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        # Print training and validation metrics
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} |f1: {f1:.4f} | AUC: {auc:.4f} | Recall: {recall:.4f}")

        # Check for early stopping
        if f1 > best_val_f1:
            best_val_f1 = f1
            early_stopping_counter = 0

            # Save the best model
            torch.save(PointnetModel.state_dict(), 'PC_12features_15_best_model_' + str(params) + '.pt')

            print("Best model saved!")

        else:
            early_stopping_counter += 1

        # Add metrics to the dataframe
        metrics_df.loc[epoch] = [epoch, auc, val_acc, f1, recall]

        # Check if early stopping criteria are met
        if early_stopping_counter >= early_stopping_patience:
            if best_val_f1 == 0:
                torch.save(PointnetModel.state_dict(), 'PC_best_model_' + str(params) + '.pt')
                # continue
            print("Early stopping!")
            break
    # testing phase for each epoch
    # running_loss = 0.0
    # y_pred1 = 0
    # total_samples = 0
    # y_true_list1 = []
    # y_pred_list1 = []
    # y_pred_prob_list1 = []
    # # test loop
    # # print("len(test",len(PC_test_dataset),len(PC_test_loader))
    # for i, data in enumerate(PC_test_loader):
    #     inputs, labels1 = data
    #     inputs, labels1 = inputs.to(device), labels1.to(device)
    #     # print("inputs",inputs,labels1)
    #     optimizer.zero_grad()
    #
    #     # Forward pass
    #     outputs: object = PointnetModel(inputs)
    #     loss = criterion(outputs, labels1)
    #
    #     # Backpropagation and optimization
    #     loss.backward()
    #     optimizer.step()
    #
    #     running_loss += loss.item()
    #     _, predicted = torch.max(outputs, 1)
    #     y_pred1 += torch.sum(predicted == labels1)
    #     total_samples += labels1.size(0)
    #     # print("y_pred",y_pred1,len(predicted),predicted,labels1)
    #     # test_correct += torch.sum(preds == labels.data)
    #
    #     # Convert labels and predictions to numpy arrays
    #     labels_np = labels1.cpu().numpy()
    #     preds_np = predicted.cpu().numpy()
    #
    #     # Append true labels and predicted probabilities to the lists
    #     y_true_list1.extend(labels_np.tolist())
    #     y_pred_list1.extend(preds_np.tolist())
    #     y_pred_prob_list1.extend(outputs[:, 1].cpu().detach().numpy().tolist())
    #
    # # Convert the lists to numpy arrays
    # y_true_test = np.array(y_true_list1)
    # y_pred_test = np.array(y_pred_list1)
    # y_pred_prob_test = np.array(y_pred_prob_list1)
    #
    # test_acc = y_pred1.double() / len(PC_test_dataset)
    # test_acc = test_acc.cpu().numpy()
    # auc = roc_auc_score(y_true_test, y_pred_prob_test)
    # f1 = f1_score(y_true_test, y_pred_test)
    # recall = recall_score(y_true_test, y_pred_test)
    # print(f"Test Acc(Epoch {epoch}): {test_acc:.4f} | AUC: {auc:.4f} | f1: {f1} | Recall: {recall:4f}")

    scheduler.step()

# Testing Accuracy
PointnetModel.eval()
running_loss = 0.0
y_pred1 = 0
total_samples = 0
y_true_list1 = []
y_pred_list1 = []
y_pred_prob_list1 = []
# test loop
with torch.no_grad():
    for i, data in enumerate(PC_test_loader):
        inputs, labels2 = data
        inputs, labels2 = inputs.to(device), labels2.to(device)
        # optimizer.zero_grad()

        # Forward pass
        outputs: object = PointnetModel(inputs)
        # loss = criterion(outputs, labels2.data)
        #
        # # Backpropagation and optimization
        # loss.backward()
        # optimizer.step()

        # running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        y_pred1 += torch.sum(predicted == labels2.data)
        total_samples += labels2.size(0)

        # Convert labels and predictions to numpy arrays
        labels_np = labels2.cpu().numpy()
        preds_np = predicted.cpu().numpy()

        # Append true labels and predicted probabilities to the lists
        y_true_list1.extend(labels_np.tolist())
        y_pred_list1.extend(preds_np.tolist())
        y_pred_prob_list1.extend(outputs[:, 1].cpu().detach().numpy().tolist())

# Convert the lists to numpy arrays
y_true_test = np.array(y_true_list1)
y_pred_test = np.array(y_pred_list1)
y_pred_prob_test = np.array(y_pred_prob_list1)

test_acc = y_pred1.double() / len(PC_test_dataset)
test_acc = test_acc.cpu().numpy()
# y_true = test_dataset.targets
# y_scores = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
auc = roc_auc_score(y_true_test, y_pred_prob_test)
# print("test",y_true_test,y_pred_test)
f1 = f1_score(y_true_test, y_pred_test)
recall = recall_score(y_true_test, y_pred_test)
# metrics_df.loc[epoch] = [epoch, auc, test_acc, f1, recall]
print(f"Test Acc: {test_acc:.4f} | AUC: {auc:.4f} | f1: {f1} | Recall: {recall:4f}")
print(params)
