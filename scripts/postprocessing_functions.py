import numpy as np


def sort_components(component_means, component_covariances):
    """
    Sorts Gaussian components based on red-channel contrast (descending).

    Args:
        component_means (np.ndarray): An array of shape (N, 3), where each row represents the mean contrast
                                      values for [blue, green, red] of a Gaussian component.
        component_covariances (np.ndarray): An array of shape (N, 3, 3), where each item is the 3x3 covariance
                                            matrix of a component.

    Returns:
        Tuple:
            - sorted_means (np.ndarray): The sorted means array by red-channel contrast (highest to lowest).
            - sorted_covariances (np.ndarray): The sorted covariance matrices in the same order.
    """
    sorted_indices = np.argsort(component_means[:, 2])[::-1]
    sorted_means = component_means[sorted_indices]
    sorted_covariances = component_covariances[sorted_indices]
    return sorted_means, sorted_covariances


def format_components(all_means_gauss, all_covariances_gauss, class_names=None):
    """
    Formats sorted Gaussian components into a dictionary with RGB contrast and covariance data.
    Automatically adds a background component with label "0" at (0,0,0).

    Args:
        all_means_gauss (np.ndarray): Array of shape (N, 3) with mean values for [blue, green, red] per component.
        all_covariances_gauss (np.ndarray): Array of shape (N, 3, 3) with covariance matrices per component.
        class_names (list, optional): List of class names for each component. If None, components are numbered.

    Returns:
        dict: A dictionary with keys as component indices (1-based) or class names and values containing:
              - 'contrast': Dict with keys 'r', 'g', 'b' for red, green, blue mean contrast.
              - 'covariance_matrix': 3x3 covariance matrix as a nested list.
    """
    component_dict = {}
    
    # Add automatic background component with label "0"
    background_covariance = [
        [0.0006, 0.0006, 0.0006],
        [0.0006, 0.0006, 0.0006],
        [0.0006, 0.0006, 0.0006]
    ]
    component_dict["0"] = {
        "contrast": {
            "r": 0.0,
            "g": 0.0,
            "b": 0.0
        },
        "covariance_matrix": background_covariance
    }

    all_means_gauss_sorted, all_covariances_gauss_sorted = sort_components(
        all_means_gauss, all_covariances_gauss
    )

    for component in range(all_means_gauss_sorted.shape[0]):
        # Use class name if provided, otherwise use component index
        if class_names and component < len(class_names):
            key = class_names[component]
        else:
            key = component + 1
            
        component_dict[key] = {}
        component_dict[key]["contrast"] = {
            "r": all_means_gauss_sorted[component][2],
            "g": all_means_gauss_sorted[component][1],
            "b": all_means_gauss_sorted[component][0],
        }
        component_dict[key][
            "covariance_matrix"
        ] = all_covariances_gauss_sorted[component].tolist()

    return component_dict
