"""
The goal of this assignment is to predict GPS coordinates from image features using k-Nearest Neighbors.
Specifically, have featurized 28616 geo-tagged images taken in Spain split into training and test sets (27.6k and 1k).

The assignment walks students through:
    * visualizing the data
    * implementing and evaluating a kNN regression model
    * analyzing model performance as a function of dataset size
    * comparing kNN against linear regression

Images were filtered from Mousselly-Sergieh et al. 2014 (https://dl.acm.org/doi/10.1145/2557642.2563673)
and scraped from Flickr in 2024. The image features were extracted using CLIP ViT-L/14@336px (https://openai.com/clip/).
"""

import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


def plot_data(train_feats, train_labels):
    """
    Input:
        train_feats: Training set image features
        train_labels: Training set GPS (lat, lon)

    Output:
        Displays plot of image locations, and first two PCA dimensions vs longitude
    """
    # Plot image locations (use marker='.' for better visibility)
    plt.scatter(train_labels[:, 1], train_labels[:, 0], marker=".")
    plt.title('Image Locations')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()

    # Run PCA on training_feats
    ##### TODO(a): Your Code Here #####
    transformed_feats = StandardScaler().fit_transform(train_feats)
    transformed_feats = PCA(n_components=2).fit_transform(transformed_feats)

    # Plot images by first two PCA dimensions (use marker='.' for better visibility)
    plt.scatter(transformed_feats[:, 0],     # Select first column
                transformed_feats[:, 1],     # Select second column
                c=train_labels[:, 1],
                marker='.')
    plt.colorbar(label='Longitude')
    plt.title('Image Features by Longitude after PCA')
    plt.show()

def mean_displacement_error(pred, actual):
    """Calculate the mean displacement error, 1 latitude = 69 miles; 1 longitude = 52 miles"""
    lat_error = np.abs(pred[:, 0] - actual[:, 0]) * 69
    lon_error = np.abs(pred[:, 1] - actual[:, 1]) * 52
    return np.mean(np.sqrt(lon_error **2 + lat_error **2))


def grid_search(train_features, train_labels, test_features, test_labels, is_weighted=False, verbose=True):
    """
    Input:
        train_features: Training set image features
        train_labels: Training set GPS (lat, lon) coords
        test_features: Test set image features
        test_labels: Test set GPS (lat, lon) coords
        is_weighted: Weight prediction by distances in feature space

    Output:
        Prints mean displacement error as a function of k
        Plots mean displacement error vs k

    Returns:
        Minimum mean displacement error
    """
    # Evaluate mean displacement error (in miles) of kNN regression for different values of k
    # Technically we are working with spherical coordinates and should be using spherical distances, but within a small
    # region like Spain we can get away with treating the coordinates as cartesian coordinates.
    knn = NearestNeighbors(n_neighbors=100).fit(train_features)

    if verbose:
        print(f'Running grid search for k (is_weighted={is_weighted})')
    ks = list(range(1, 11)) + [20, 30, 40, 50, 100]
    mean_errors = []
    for k in ks:
        # dist: knn from each test point (test points x k); indices: of knn from each test point (test points x k)
        distances, indices = knn.kneighbors(test_features, n_neighbors=k)

        errors = []
        # i is which test point, nearest is the k nearest
        for i, nearest in enumerate(indices):
            # Evaluate mean displacement error in miles for each test image
            # Assume 1 degree latitude is 69 miles and 1 degree longitude is 52 miles
            y = test_labels[i]

            ##### (d): Your Code Here #####
            if not is_weighted: 
                pred_label = np.mean(train_labels[nearest], axis=0)
            ##### (f): Your Code Here #####
            else:
                # Weighted mean of labels of k nearest neighbors
                weights = 1 / (distances[i] + 1e-8) 
                weighted_labels = (
                    train_labels[nearest] * weights[:, None]
                )  # convert weights into 2d: n * 1
                pred_label = np.sum(weighted_labels, axis=0) / np.sum(weights)

            mde = mean_displacement_error(np.array([pred_label]), np.array([y]))
            errors.append(mde)

        e = np.mean(np.array(errors))
        mean_errors.append(e)
        if verbose:
            print(f'{k}-NN mean displacement error (miles): {e:.1f}')

    # Plot error vs k for k Nearest Neighbors
    if verbose:
        plt.plot(ks, mean_errors)
        plt.xlabel('k')
        plt.ylabel('Mean Displacement Error (miles)')
        plt.title('Mean Displacement Error (miles) vs. k in kNN')
        plt.show()

    return min(mean_errors)


def main():
    print("Predicting GPS from CLIP image features\n")

    # Import Data
    print("Loading Data")
    data = np.load('im2spain_data.npz')

    train_features = data['train_features']  # [N_train, dim] array
    test_features = data['test_features']    # [N_test, dim] array
    train_labels = data['train_labels']      # [N_train, 2] array of (lat, lon) coords
    test_labels = data['test_labels']        # [N_test, 2] array of (lat, lon) coords
    train_files = data['train_files']        # [N_train] array of strings
    test_files = data['test_files']          # [N_test] array of strings

    # Data Information
    print('Train Data Count:', train_features.shape[0])

    # Part A: Feature and label visualization (modify plot_data method)
    # plot_data(train_features, train_labels)

    # Part B: Find the 3 nearest neighbors of test image 53633239060.jpg
    knn = NearestNeighbors(n_neighbors=3).fit(train_features)

    # Use knn to get the k nearest neighbors of the features of image 53633239060.jpg
    ##### (b): Your Code Here #####
    test_img_idx = np.where(test_files == "53633239060.jpg")[0][0] # get index from nested array 
    test_img_features = test_features[test_img_idx]
    print(f"the correct coordinates are {test_labels[test_img_idx]}")
    nn3_distances, nn3_indices = knn.kneighbors([test_img_features])

    # get 3 nearest neighbors
    print("\nThree nearest neighbors in the training set for image 53633239060.jpg:")
    for i, idx in enumerate(nn3_indices[0]):  # access nested array
        neighbor_file = train_files[idx]
        neighbor_coords = train_labels[idx]
        print(f"{i+1}: File: {neighbor_file}, Coordinates: {neighbor_coords}")

    # Part C: establish a naive baseline of predicting the mean of the training set
    ##### TODO(c): Your Code Here #####
    mean_lat = np.mean(train_labels[:, 0])
    mean_lon = np.mean(train_labels[:, 1])
    centroid = np.array([mean_lat, mean_lon])
    print(f"Training set centroid (mean GPS coordinates): Latitude = {mean_lat}, Longitude = {mean_lon}")

    num_test_images = test_labels.shape[0]
    pred_labels = np.tile(
        centroid, (num_test_images, 1)
    )  # to have centroid repeated num times ver and 1 time hori
    mde = mean_displacement_error(pred_labels, test_labels)
    print(f"Constant Baseline MDE (centroid) is: {mde} miles")

    # Part D or E: complete grid_search to find the best value of k
    grid_search(train_features, train_labels, test_features, test_labels)

    # Parts F: rerun grid search after modifications to find the best value of k
    grid_search(train_features, train_labels, test_features, test_labels, is_weighted=True)

    # Part G: compare to linear regression for different # of training points
    mean_errors_lin = []
    mean_errors_nn = []
    ratios = np.arange(0.1, 1.1, 0.1)
    for r in ratios:
        num_samples = int(r * len(train_features))
        ##### TODO(g): Your Code Here #####
        shuffled_indices = np.random.permutation(len(train_features))
        subset_indices = shuffled_indices[:num_samples]
        subset_train_features = train_features[subset_indices]
        subset_train_labels = train_labels[subset_indices]

        # get mde for knn
        print("\nRunning grid search for weighted k-NN...")
        e_nn = grid_search(
            subset_train_features,
            subset_train_labels,
            test_features,
            test_labels,
            is_weighted=True,
        )

        # get mde for lin regression
        print("\nEvaluating Linear Regression...")
        lin_reg = LinearRegression()
        lin_reg.fit(subset_train_features, subset_train_labels)
        pred_labels_lin_reg = lin_reg.predict(test_features)
        e_lin = mean_displacement_error(pred_labels_lin_reg, test_labels)

        mean_errors_nn.append(e_nn)
        mean_errors_lin.append(e_lin)

        print(f'\nTraining set ratio: {r} ({num_samples})')
        print(f'Linear Regression mean displacement error (miles): {e_lin:.1f}')
        print(f'kNN mean displacement error (miles): {e_nn:.1f}')

    # Plot error vs training set size
    plt.plot(ratios, mean_errors_lin, label='lin. reg.')
    plt.plot(ratios, mean_errors_nn, label='kNN')
    plt.xlabel('Training Set Ratio')
    plt.ylabel('Mean Displacement Error (miles)')
    plt.title('Mean Displacement Error (miles) vs. Training Set Ratio')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
