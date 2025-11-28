import cv2
import numpy as np
import os
import time
import numba

@numba.njit(parallel=True)
def get_contrasts_from_dir(image_directory, mask_directory, flatfield_path):
    full_colors = []
    full_contrasts = []
    background_colors = []

    flatfield = cv2.imread(flatfield_path)
    assert (
        flatfield is not None
    ), "Could not load flatfield, have you selected the correct path?"

    mask_names = os.listdir(mask_directory)

    for idx, image_name in enumerate(mask_names):

        image_path = os.path.join(image_directory, image_name)
        mask_path = os.path.join(mask_directory, image_name)

        print(f"{idx + 1}/{len(mask_names)} read", end="\r")

        mask = cv2.imread(mask_path, 0)
        image = cv2.imread(image_path)

        mask = cv2.erode(
            mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2
        )
        if cv2.countNonZero(mask) < 10:
            continue

        image = remove_vignette(image, flatfield)
        flake_color = np.array(image[mask == 255])
        background_color = np.array(calculate_background_color(image, 5))

        if np.any(background_color == 0):
            print(f"Error with image {image_name}, skipping")
            continue

        flake_contrast = (flake_color / background_color) - 1

        background_colors.append(background_color)
        full_colors.extend(flake_color)
        full_contrasts.extend(flake_contrast)

    full_colors = np.array(full_colors)
    background_colors = np.array(background_colors)
    full_contrasts = np.array(full_contrasts)

    return full_colors, full_contrasts, background_colors


def remove_vignette_legecy(image, flat_field):
    """Removes the Vignette from the Image

    Args:
        image (NxMx3 Array): The Image with the Vignette
        flat_field (NxMx3 Array): the Flat Field in RGB

    Returns:
        (NxMx3 Array): The Image without the Vignette
    """
    # convert to hsv and cast to 16bit, to be able to add more than 255
    image_hsv = np.asarray(cv2.cvtColor(image, cv2.COLOR_BGR2HSV), dtype=np.uint16)
    flat_field_hsv = np.asarray(
        cv2.cvtColor(flat_field, cv2.COLOR_BGR2HSV), dtype=np.uint16
    )

    # get the filter and apply it to the image
    image_hsv[:, :, 2] = (
        image_hsv[:, :, 2]
        / flat_field_hsv[:, :, 2]
        * cv2.mean(flat_field_hsv[:, :, 2])[0]
    )

    # clip it back to 255
    image_hsv[:, :, 2][image_hsv[:, :, 2] > 255] = 255

    # Recast to uint8 as the color depth is 8bit per channel
    image_hsv = np.asarray(image_hsv, dtype=np.uint8)

    # reconvert to bgr
    image_no_vigentte = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)
    return image_no_vigentte


def remove_vignette(
    image,
    flat_field,
    max_background_value: int = 241,
):
    """Removes the Vignette from the Image

    Args:
        image (NxMx3 Array): The Image with the Vignette
        flat_field (NxMx3 Array): the Flat Field in RGB
        max_background_value (int): the maximum value of the background

    Returns:
        (NxMx3 Array): The Image without the Vignette
    """

    image_no_vigentte = image / flat_field * cv2.mean(flat_field)[:-1]

    image_no_vigentte[image_no_vigentte > max_background_value] = max_background_value

    return np.asarray(image_no_vigentte, dtype=np.uint8)


def calculate_background_color(img, radius=2):

    masks = []

    for i in range(3):
        img_channel = img[:, :, i]

        # A threshold which removes the Unwanted background of the non chip
        # Currently Unused

        mask = cv2.inRange(img_channel, 0, 230)

        hist_r = cv2.calcHist([img_channel], [0], mask, [256], [0, 256])

        hist_max_r = np.argmax(hist_r)

        threshed_r = cv2.inRange(
            img_channel, int(hist_max_r - radius), int(hist_max_r + radius)
        )
        background_mask_channel = cv2.erode(threshed_r, np.ones((3, 3)), iterations=3)
        masks.append(background_mask_channel)

    final_mask = cv2.bitwise_and(masks[0], masks[1])
    final_mask = cv2.bitwise_and(final_mask, masks[2])

    return cv2.mean(img, mask=final_mask)[:3]

def get_instance_contrasts_from_dir(
    image_directory,
    mask_directory,
    flatfield_path=None,
    use_flatfield=False,
    min_instance_size=1,
):
    """
    Extracts contrast-normalized pixel values for each instance in a set of images and corresponding masks.
    For each mask in the mask_directory, the function finds the corresponding image in image_directory,
    optionally applies flatfield correction, computes the background color, and calculates the contrast image.
    For each instance (connected component) in the mask (excluding background), it collects the pixel values
    from the contrast image, provided the instance is larger than `min_instance_size`.
    Args:
        image_directory (str): Path to the directory containing input images.
        mask_directory (str): Path to the directory containing instance masks (grayscale images).
        flatfield_path (str, optional): Path to the flatfield image for vignette correction. Defaults to None.
        use_flatfield (bool, optional): Whether to apply flatfield correction. Defaults to False.
        min_instance_size (int, optional): Minimum number of pixels for an instance to be considered. Defaults to 10.
    Returns:
        Tuple[List[np.ndarray], List[Tuple[str, int]]]:
            - instance_contrasts: List of N x 3 arrays, where each array contains the BGR contrast values for all pixels in an instance.
            - instance_classifiers: List of tuples (mask_name, instance_id) identifying each instance.
    """
    # returns a list of all instances
    # each instance is a N x 3 Array with N being the number of pixels in the instance and 3 being the BGR values
    instance_contrasts = []
    instance_classifiers = []

    if use_flatfield and flatfield_path is not None:
        flatfield = cv2.imread(flatfield_path)
        assert (
            flatfield is not None
        ), f"Could not load flatfield at '{flatfield_path}', have you selected the correct path?"

    mask_names = os.listdir(mask_directory)

    for idx, mask_name in enumerate(mask_names):
        print(f"{idx + 1}/{len(mask_names)}", end="\n")

        image_path = os.path.join(image_directory, mask_name)
        if not os.path.exists(image_path):
            image_path = os.path.join(
                image_directory, mask_name.replace(".png", ".jpg")
            )

        if not os.path.exists(image_path):
            print(f"Could not find image corresponding to mask '{mask_name}', skipping")
            continue

        mask_path = os.path.join(mask_directory, mask_name)

        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        image = cv2.imread(image_path)

        assert mask is not None, f"Could not load mask {mask_path}"
        assert image is not None, f"Could not load image {image_path}"

        if use_flatfield and flatfield_path is not None:
            image = remove_vignette(image, flatfield)

        background_color = calculate_background_color(image)

        if np.any(background_color == 0):
            print(f"Error with image {mask_name}; Invalid Background, skipping")
            continue

        contrast_image = image / background_color - 1
        
        for instance_id in np.unique(mask):
            # skipping the background
            if instance_id == 0:
                continue

            instance_mask = np.zeros_like(mask)
            instance_mask[mask == instance_id] = 255

            if cv2.countNonZero(instance_mask) < min_instance_size:
                continue

            instance_contrasts.append(contrast_image[instance_mask == 255])
            instance_classifiers.append((mask_name, instance_id))

    return instance_contrasts, instance_classifiers