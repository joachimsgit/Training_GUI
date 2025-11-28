import cv2
import numpy as np
from scipy import ndimage
from skimage.measure import label
import os

def binary_to_instance_mask(binary_mask):
    """
    Convert a binary mask to an instance mask where each connected component
    gets a unique ID.
    
    Args:
        binary_mask (np.ndarray): Binary mask with values 0 (background) and 255 (foreground)
        
    Returns:
        np.ndarray: Instance mask with unique IDs for each connected component
    """
    # Ensure binary mask is in correct format (0 and 1)
    if binary_mask.max() > 1:
        binary_mask = (binary_mask > 0).astype(np.uint8)
    
    # Label connected components
    # connectivity=2 means 8-connectivity (includes diagonal neighbors)
    labeled_mask, num_features = label(binary_mask, connectivity=2, return_num=True)

    
    return labeled_mask.astype(np.uint16)  # Use uint16 to support more instances

def instance_to_binary_mask(instance_mask):
    """
    Convert an instance mask to a binary mask.
    
    Args:
        instance_mask (np.ndarray): Instance mask with unique IDs for each instance
        
    Returns:
        np.ndarray: Binary mask with 255 for foreground, 0 for background
    """
    # Create binary mask: any non-zero value becomes 255
    binary_mask = (instance_mask > 0).astype(np.uint8) * 255
    
    num_instances = len(np.unique(instance_mask)) - 1  # Subtract 1 for background
    
    return binary_mask

def filter_instance_mask_by_size(instance_mask, min_area=50, max_area=None):
    """
    Filter instance mask by removing instances that are too small or too large.
    
    Args:
        instance_mask (np.ndarray): Instance mask with unique IDs
        min_area (int): Minimum area for an instance to be kept
        max_area (int, optional): Maximum area for an instance to be kept
        
    Returns:
        np.ndarray: Filtered instance mask with renumbered instances
    """
    filtered_mask = np.zeros_like(instance_mask)
    new_id = 1
    
    unique_ids = np.unique(instance_mask)
    unique_ids = unique_ids[unique_ids > 0]  # Remove background (0)
    
    for instance_id in unique_ids:
        instance_area = np.sum(instance_mask == instance_id)
        
        # Check size constraints
        if instance_area >= min_area:
            if max_area is None or instance_area <= max_area:
                filtered_mask[instance_mask == instance_id] = new_id
                new_id += 1
    
    removed_count = len(unique_ids) - (new_id - 1)
    
    return filtered_mask


# Example usage functions
def convert_watershed_mask_to_instance(watershed_mask_path, output_path):
    """
    Convert a watershed binary mask to an instance mask and save it.
    
    Args:
        watershed_mask_path (str): Path to binary watershed mask
        output_path (str): Path to save instance mask
    """
    # Load binary mask
    binary_mask = cv2.imread(watershed_mask_path, cv2.IMREAD_GRAYSCALE)
    
    # Convert to instance mask
    instance_mask = binary_to_instance_mask(binary_mask)
    
    # Filter small instances (optional)
    instance_mask = filter_instance_mask_by_size(instance_mask, min_area=50)
    
    # Save as 16-bit PNG to preserve instance IDs
    cv2.imwrite(output_path, instance_mask.astype(np.uint16))

def convert_instance_mask_to_binary(instance_mask_path, output_path):
    """
    Convert an instance mask to a binary mask and save it.
    
    Args:
        instance_mask_path (str): Path to instance mask
        output_path (str): Path to save binary mask
    """
    # Load instance mask
    instance_mask = cv2.imread(instance_mask_path, cv2.IMREAD_UNCHANGED)
    
    # Convert to binary mask
    binary_mask = instance_to_binary_mask(instance_mask)
    
    # Save binary mask
    cv2.imwrite(output_path, binary_mask)
    
    print(f"Saved binary mask to {output_path}")

def convert_semantic_mask_to_instance(semantic_mask_path, output_path):
    """
    Convert a semantic mask to an instance mask by labeling connected components.
    
    Args:
        semantic_mask_path (str): Path to semantic mask
        output_path (str): Path to save instance mask
    """
    # Load semantic mask
    semantic_mask = cv2.imread(semantic_mask_path, cv2.IMREAD_GRAYSCALE)
    
    # Convert to instance mask
    instance_mask = binary_to_instance_mask(semantic_mask)
    
    # Filter small instances (optional)
    instance_mask = filter_instance_mask_by_size(instance_mask, min_area=50)
    
    # Save as 16-bit PNG to preserve instance IDs
    cv2.imwrite(output_path, instance_mask.astype(np.uint16))
    
    print(f"Saved instance mask to {output_path}")

if __name__ == "__main__":
    semantic_mask_dir = "Materials/WSe2/semantic_masks"
    instance_output_dir = "Materials/WSe2/masks"  # Fixed: should be "masks" not "instance_masks"
    os.makedirs(instance_output_dir, exist_ok=True)

    semantic_masks = [f for f in os.listdir(semantic_mask_dir) if f.endswith('.png')]

    for semantic_mask in semantic_masks:
        semantic_mask_path = os.path.join(semantic_mask_dir, semantic_mask)
        instance_mask_path = os.path.join(instance_output_dir, semantic_mask.replace('semantic', 'instance'))

        convert_semantic_mask_to_instance(semantic_mask_path, instance_mask_path)
        print(f"Converted {semantic_mask_path} to {instance_mask_path}")
