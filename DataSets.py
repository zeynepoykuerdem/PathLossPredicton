import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sci
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor


grid_size = 500
base_station_frequence = 1800  # mhz especially for COST 231
base_station = (250, 250)
base_station_height = 30  # m
base_station_transmit_power = 20  # dbm

# receivers
numberOfRecevivers = 50
receivers_height = 10  # m
receiver_positions = np.random.randint(0, grid_size, size=(numberOfRecevivers, 2))

# buildings
numberOfBuildings = 30
building_heigths = [10, 30]  # 10m or 30m
building_widths = [10, 30]
building_lengths = [15, 50]

# trees
numberOfTrees = 100
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
    return distance / 1000  # km


# DataSets
def calculateFreeSpaceLoss(distance):
    # FSPL= 20 log 10(dkm)+ 20 log 10 (frequency ghz)+ 20 log 10 (4pi/speed of light(m/s)-Gtx-gRx
    return 20 * np.log10(distance) + 20 * np.log10(base_station_frequence) + 20 * np.log10(4 * np.pi / sci.constants.c)


def cost231_HataModel(distance, building_density, tree_density, Cm=0, f=base_station_frequence, hm=receivers_height):
    #
    term1 = 46.3 + 33.9 * np.log10(f) - 13.82 * np.log10(base_station_height) - (
            (1.1 * np.log10(f) - 0.7) * hm - (1.56 * np.log10(f) - 0.8))
    term2 = (44.9 - 6.55 * np.log10(base_station_height)) * np.log10(distance) + Cm
    path_loss = term1 + term2 + building_density + tree_density
    return path_loss


def building_loss(receiver_height):
    for height in building_heigths:
        if receiver_height <= height:
            building_loss = height * 0.1
    return building_loss


def tree_loss(receiver_height):
    for height in tree_heights:
        if receiver_height <= height:
            tree_loss = height * 0.05
    return tree_loss


def building_density(receiverPosition):
    # check if the receiver is within the range of a building
    building_density = 0
    total_building_height = 0
    for position, height in zip(building_positions, building_heigths):

        buildingDistance = np.linalg.norm(np.array(receiverPosition) - np.array(position))
        if buildingDistance < 500:
            building_density += 1

    return building_density * 0.1


def tree_dencity(receiverPosition):
    tree_dencity = 0
    for position in tree_positions:
        treeDistance = np.linalg.norm(np.array(receiverPosition) - np.array(position))
        if treeDistance < 200:
            tree_dencity += 1
    # check if the receiver is within the range of a building

    return tree_dencity * 0.05


data_set = []
distances = []
path_loss_Cost231 = []
tree_loss_array = []
building_loss_array = []

for receiver in receivers_positions:
    # receiver 1 (x,y)
    distance = calculateDistanceBetweenReceiverBase(base_station, receiver)
    distances.append(distance)

    building_density_f = building_density(receiver)
    tree_density_f = tree_dencity(receiver)
    pl_cost231 = cost231_HataModel(distance, building_density=building_density_f, tree_density=tree_density_f)
    path_loss_Cost231.append(pl_cost231)

    free_space_loss = calculateFreeSpaceLoss(distance)

    tree_density = tree_dencity(receiver)

    data_set.append({
        "Receiver_X": receiver[0],
        "Receiver_Y": receiver[1],
        "Distance_to_Base_Station (m)": distance,
        "Transmitter Height": base_station_height,
        "Receiver Height": receivers_height,
        "Base_Station_Frequency (GHz)": base_station_frequence,
        "Path_Loss_COST_231 (dB)": pl_cost231

    })

# Creating a DataFrame
df = pd.DataFrame(data_set)
print(df.head(10))

# Saved as CSV file
df.to_csv("path_loss_data.csv", index=False)
plt.figure(figsize=(10, 8))
plt.plot(distances, path_loss_Cost231, color="blue", label='Cost 231')
plt.xlabel("Distance to Base Station (m)")
plt.ylabel("Path Loss (dB)")
plt.show()



# preparing data for Machine Learning Algorithms
# X=> the features are :distanceBetweenReceiverandBaseStation, building_loss, tree_loss
# Y->PathLoss

with open("path_loss_data.csv", 'r', encoding='iso-8859-1') as f:
    data_set = list(csv.reader(f))

x1 = np.array([])
x2 = np.array([])
x3 = np.array([])
x4 = np.array([])
x5 = np.array([])
x6 = np.array([])
y = np.array([])

for row in data_set[1:]:
    x1 = np.append(x1, float(row[2]))
    x2 = np.append(x2, float(row[3]))
    x3 = np.append(x3, float(row[4]))
    x4 = np.append(x4, float(row[5]))
    y = np.append(y, float(row[6]))

X = np.column_stack((x1, x2, x3, x4))
sc = StandardScaler()
X_scaled = sc.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)




'''Random Forest Regressor Machine Learning Model'''
# Random Forest Block Diagram

param_grid = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None, ],
    'max_leaf_nodes': [5, 10, 20, None]
}
random_search = RandomizedSearchCV(estimator=RandomForestRegressor(), param_distributions=param_grid, n_iter=10, cv=3,
                                   verbose=2, random_state=42, n_jobs=-1)

random_search.fit(X_train, y_train)

print("Best Parameters:", random_search.best_params_)
best_rf = random_search.best_estimator_

# randomForestRegressor = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10, max_leaf_nodes=5,min_samples_split=5)
# randomForestRegressor.fit(X_train, y_train)

y_train_pred = best_rf.predict(X_train)
train_mse = mean_squared_error(y_train, y_train_pred)
train_rf2 = r2_score(y_train, y_train_pred)

y_pred = best_rf.predict(X_test)
test_mse_rf = mean_squared_error(y_test, y_pred)
r2_rf = r2_score(y_test, y_pred)

print(f"Test Mean Squared Error of Random Forest: {test_mse_rf}")
print(f"Train Mean Squared Error of Random Forest: {train_mse}")
print(f"Train R2 Score of Random Forest: {train_rf2}")
print(f" TestR2 Score of Random Forest: {r2_rf}")

''' SVR Model'''
# SVR Block Diagram

param = {'C': [0.1, 1],
         'gamma': [0.01, 0.001, 0.0001], 'kernel': ['rbf', 'linear']
         }
# C is a regularization parameter, gamma is a coefficient for 'rbf,'poly','sigmoid
# svr=SVR(kernel='linear')#rbf:radial basis function it is not related with our task because it is not a complex patterns
random_search1 = RandomizedSearchCV(estimator=SVR(), param_distributions=param, n_iter=10, cv=3, verbose=2,
                                    random_state=42, n_jobs=-1)
random_search1.fit(X_train, y_train)
print("Best Parameters for SVR:", random_search1.best_params_)
best_svr = random_search1.best_estimator_
svr_y_pred = best_svr.predict(X_test)

# testing training mse
svr_y_train_pred = best_rf.predict(X_train)
svr_train_mse = mean_squared_error(y_train, svr_y_train_pred)

svr_mse = mean_squared_error(y_test, svr_y_pred)
svr_r2 = r2_score(y_test, svr_y_pred)


print(f"Train Mean Squared Error of SVR: {svr_train_mse}")
print(f"Test Mean Squared Error of SVR: {svr_mse}")
print(f"R2 of SVR: {svr_r2}")

''' Multilayer Perceptron Regressor '''

# performing feature scaling to help us to normalize the data
# Building an Multilayer Perceptron Regressor
# inputlayers->hiddenlayers->outputlayers
# sigmoid function, feedforwardfunc,backpropogation

model=MLPRegressor(max_iter=500,activation='relu',solver='adam',learning_rate_init=0.01,hidden_layer_sizes=(5,5),alpha=0.01)


model.fit(X_train,y_train)

y_pred_ann=model.predict(X_test)

y_pred_train= model.predict(X_train)

train_mse_ann=mean_squared_error(y_train,y_pred_train)
mse=mean_squared_error(y_test,y_pred_ann)
r2=r2_score(y_test,y_pred_ann)
r2_train=r2_score(y_train,y_pred_train)

print(f"Test Mean Squared Error (MSE): {mse}")
print(f"Train Mean Squared Error (MSE): {train_mse_ann}")
print(f"R-squared (R²): {r2}")
print(f" Train R-squared (R²): {r2_train}")

models=["Random Forrest","SVR","MLP"]
mse_train=[train_mse,svr_train_mse,train_mse_ann]
mse_test=[test_mse_rf,svr_mse,mse]
r2_scores=[r2_rf,svr_r2,r2]

fig,ax1= plt.subplots(figsize=(10,6))

ax1.plot(models,mse_train,label="Train MSE",alpha=0.6,color='blue')
ax1.plot(models, mse_test, label="Test MSE",alpha=0.6,color='green')
ax1.set_xlabel('Models')
ax1.set_ylabel('MSE',color='blue')
ax1.tick_params(axis='y',labelcolor='red')
ax1.set_title('Comparison of MSE and R² for SVR Models')
ax2 = ax1.twinx()
ax2.plot(models, r2_scores, color='tab:red', marker='o', label="R² Score", linewidth=2)
ax2.set_ylabel('R² Score', color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.show()