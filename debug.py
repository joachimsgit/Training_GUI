import numpy as np
import cv2
from zpreprocessor_functions import get_instance_contrasts_from_dir

INSTANCE_MASK_DIRECTORY = "Materials/WS2/masks"
IMAGE_DIRECTORY = "Materials/WS2/images"




def check_for_instances(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    unique_values = np.unique(mask)
    print(f"Unique values in mask: {unique_values}")
    return len(unique_values) > 1  

num_instances = check_for_instances(f"{INSTANCE_MASK_DIRECTORY}/1.png")
num_instances = check_for_instances(f"{INSTANCE_MASK_DIRECTORY}/2.png")
num_instances = check_for_instances(f"{INSTANCE_MASK_DIRECTORY}/3.png")
num_instances = check_for_instances(f"{INSTANCE_MASK_DIRECTORY}/4.png")

instance_contrast, instance_classifiers = get_instance_contrasts_from_dir(
    IMAGE_DIRECTORY,
    INSTANCE_MASK_DIRECTORY,
    use_flatfield=False,
    flatfield_path=None,
    min_instance_size=1,
)
