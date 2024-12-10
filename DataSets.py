import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sci
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV

grid_size = 2000
base_station_frequence = 2.0  # Ghz especially for COST 231
base_station = (1000, 1000)
base_station_height = 20  # m
base_station_transmit_power = 20  # dbm

# receivers
numberOfRecevivers = 50
receivers_height = 10  # m
receiver_positions = np.random.randint(0, grid_size, size=(numberOfRecevivers, 2))

# buildings
numberOfBuildings = 100
building_heigths = [10, 30]  # 10m or 30m
building_widths = [10, 30]
building_lengths = [15, 50]

# trees
numberOfTrees = 50
tree_heights = [2, 8]

np.random.seed(42)
# terrain is not flat in this scenario
terrain_heights = np.random.randint(0, 2, size=(grid_size, grid_size))
building_positions = np.random.randint(50, grid_size - 50, size=(numberOfBuildings, 2))
tree_positions = np.random.randint(80, grid_size - 80, size=(numberOfTrees, 2))
building_widths_array = np.random.choice(building_widths, numberOfBuildings)
building_heigths_array = np.random.choice(building_heigths, numberOfBuildings)
building_lengths_array = np.random.choice(building_lengths, numberOfBuildings)
receivers_positions = np.random.randint(0, grid_size, size=(numberOfRecevivers, 2))

# Plotting the terrain
plt.figure(figsize=(10, 8))
sns.heatmap(terrain_heights, cmap="terrain", cbar_kws={'label': 'Height (meters)'})
plt.title("Synthetic Urban Area Terrain")
plt.xlabel("X (meters)")
plt.ylabel("Y (meters)")
plt.show()

# plotting the synthetic environment
plt.figure(figsize=(8, 8))
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.gca().set_aspect('equal', adjustable='box')
plt.xlim(0, grid_size)
plt.ylim(0, grid_size)
plt.xlabel("X (meters)")
plt.ylabel("Y (meters)")

# Plot buildings
for position, width, length, height in zip(building_positions, building_lengths_array, building_widths_array,
                                           building_heigths_array):
    if height == 10:
        plt.gca().add_patch(plt.Rectangle(position, width, length, color='purple', alpha=0.7,
                                          label=f"H:{height}m,W:{width}m,L:{length}m"))
    else:
        plt.gca().add_patch(plt.Rectangle(position, width, length, color='pink', alpha=0.7,
                                          label=f"H:{height}m,W:{width}m,L:{length}m"))

# Plot Trees
plt.scatter(tree_positions[:, 0], tree_positions[:, 1], c='r', label='Trees', zorder=5)
# Plot receivers
plt.scatter(receivers_positions[:, 0], receivers_positions[:, 1], c='blue', label='Receivers', zorder=5)

# Plot base station
plt.scatter(*base_station, c='green', label='Base Station', s=100, zorder=5)

# Add legend
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc="upper left")

# Title and display
plt.title("Synthetic Urban Environment")
plt.show()


def calculateDistanceBetweenReceiverBase(base_station, receiver_position):
    distance = np.sqrt((receiver_position[0] - base_station[0]) ** 2 + (receiver_position[1] - base_station[1]) ** 2)
    return distance


# DataSets
def calculateFreeSpaceLoss(distance):
    # FSPL= 20 log 10(dkm)+ 20 log 10 (frequency ghz)+ 20 log 10 (4pi/speed of light(m/s)-Gtx-gRx
    return 20 * np.log10(distance) + 20 * np.log10(base_station_frequence) + 20 * np.log10(4 * np.pi / sci.constants.c)


def cost231_model(distance, f=base_station_frequence, hm=receivers_height, Cm=0):
    term1 = 46.3 + 33.9 * np.log10(f) - 13.82 * np.log10(base_station_height) - hm
    term2 = (44.9 - 6.55 * np.log10(base_station_height)) * np.log10(distance)
    path_loss = term1 + term2 + Cm

    return path_loss


# her receiver and base station icin ayri path loss bakilmasi lazim

def building_loss(receiverPosition):
    # check if the receiver is within the range of a building
    building_loss = 0
    for position, height, width, lenght in zip(building_positions, building_heigths, building_widths, building_lengths):
        # check if the receiver is within the range of a building
        buildingDistance = np.linalg.norm(np.array(receiverPosition) - np.array(position))
        if buildingDistance < 500:
            print(f"Receiver: {receiverPosition}, Building: {position}, Distance: {buildingDistance}")
            building_loss += height * 0.1

    return building_loss


def tree_loss(receiverPosition):
    tree_loss = 0
    for position, height in zip(tree_positions, tree_heights):
        treeDistance = np.linalg.norm(np.array(receiverPosition) - np.array(position))
        if treeDistance < 200:
            print(f"Receiver: {receiverPosition}, Tree: {position}, Distance: {treeDistance}")
            tree_loss += height * 0.05
    # check if the receiver is within the range of a building

    return tree_loss


data_set = []
distances = []
path_losses_with_obstacles = []
tree_loss_array = []
building_loss_array = []

for receiver in receivers_positions:
    # receiver 1 (x,y)
    distance = calculateDistanceBetweenReceiverBase(base_station, receiver)
    distances.append(distance)
    free_space_loss = calculateFreeSpaceLoss(distance)
    pl_cost231_without_obstacles = cost231_model(distance)
    building_loss_f = building_loss(receiver)
    building_loss_array.append(building_loss_f)
    tree_loss_f = tree_loss(receiver)
    tree_loss_array.append(tree_loss_f)
    pl_cost231 = pl_cost231_without_obstacles + building_loss_f + tree_loss_f
    path_losses_with_obstacles.append(pl_cost231)

    data_set.append({
        "Receiver_X": receiver[0],
        "Receiver_Y": receiver[1],
        "Distance_to_Base_Station (m)": distance,
        "Free_Space_Path_Loss (dB)": free_space_loss,
        "Building Loss (dB)": building_loss_f,
        "Tree Loss(dB)": tree_loss_f,
        "Path_Loss_COST_231 (dB)": pl_cost231,
        "Base_Station_Frequency (GHz)": base_station_frequence
    })

# Creating a DataFrame
df = pd.DataFrame(data_set)
print(df.head(10))

# Saved as CSV file
df.to_csv("path_loss_data.csv", index=False)
# Graph to illustrate to illustrate PL according the distance and obstacles
plt.figure(figsize=(10, 8))
plt.plot(distances, path_losses_with_obstacles, color="blue", label='Cost 231')
plt.scatter(distances, path_losses_with_obstacles, color="red", alpha=0.6, label="Receivers", s=10)
plt.xlabel("Distance to Base Station (m)")
plt.ylabel("Path Loss (dB)")
plt.show()

# Data Normalization because the Data Sets it is not in Gaussian Form

data = pd.read_csv("path_loss_data.csv")
scaler = MinMaxScaler()
normalized_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns, index=data.index)
normalized_data.to_csv("normalized_data_set.csv", index=False)
# now we have a normalized data in which evaluating the algorithm performance later

# preparing data for Machine Learning Algorithms
# X=> the features are :distanceBetweenReceiverandBaseStation, building_loss, tree_loss
# Y->PathLoss

data = pd.read_csv("normalized_data_set.csv")

x1 = np.array([])
x2 = np.array([])
x3 = np.array([])
y = np.array([])

for row in data[1:51]:
    x1 = np.append(x1, float(row[2]))
    x2 = np.append(x2, float(row[4]))
    x3 = np.append(x3, float(row[5]))
    y = np.append(y, float(row[6]))

X = np.column_stack((x1, x2, x3))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
