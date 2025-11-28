import cv2
import json
import numpy as np
import os
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import pycocotools.mask as mask_util
from PIL import Image


class COCOConverter:
    """
    Convert instance masks to COCO format annotations for Mask2Former training.
    
    Takes directory of instance masks (16-bit PNG with unique IDs) and converts
    to COCO JSON format with proper annotations, categories, and metadata.
    """
    
    def __init__(self, dataset_name: str = "custom_dataset", dataset_description: str = ""):
        """
        Initialize COCO converter.
        
        Args:
            dataset_name: Name of the dataset
            dataset_description: Description of the dataset
        """
        self.dataset_name = dataset_name
        self.dataset_description = dataset_description
        self.reset_annotations()
        
    def reset_annotations(self):
        """Reset all annotation data for new conversion."""
        self.coco_data = {
            "info": {
                "description": self.dataset_description,
                "version": "1.0",
                "year": datetime.now().year,
                "contributor": "GMM Training GUI",
                "date_created": datetime.now().isoformat(),
                "url": ""
            },
            "licenses": [
                {
                    "id": 1,
                    "name": "Custom License",
                    "url": ""
                }
            ],
            "categories": [],
            "images": [],
            "annotations": []
        }
        self.image_id_counter = 1
        self.annotation_id_counter = 1
        self.category_id_map = {}
        
    def add_category(self, category_id: int, category_name: str, supercategory: str = "object"):
        """
        Add a category to the COCO dataset.
        
        Args:
            category_id: Unique ID for the category
            category_name: Human-readable name for the category
            supercategory: Parent category name
        """
        if category_id not in self.category_id_map:
            self.category_id_map[category_id] = len(self.coco_data["categories"]) + 1
            self.coco_data["categories"].append({
                "id": self.category_id_map[category_id],
                "name": category_name,
                "supercategory": supercategory
            })
    
    def mask_to_rle(self, mask: np.ndarray) -> Dict:
        """
        Convert binary mask to RLE (Run Length Encoding) format.
        
        Args:
            mask: Binary mask (2D numpy array)
            
        Returns:
            RLE encoded mask dictionary
        """
        # Convert to uint8 and ensure binary (0 or 1)
        binary_mask = (mask > 0).astype(np.uint8)
        
        # Fortran order for pycocotools compatibility
        fortran_mask = np.asfortranarray(binary_mask)
        
        # Encode to RLE
        rle = mask_util.encode(fortran_mask)
        
        # Convert bytes to string for JSON serialization
        if isinstance(rle['counts'], bytes):
            rle['counts'] = rle['counts'].decode('utf-8')
            
        return rle
    
    def get_bbox_from_mask(self, mask: np.ndarray) -> List[int]:
        """
        Get bounding box from binary mask in COCO format [x, y, width, height].
        
        Args:
            mask: Binary mask (2D numpy array)
            
        Returns:
            Bounding box as [x, y, width, height]
        """
        # Find non-zero pixels
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            return [0, 0, 0, 0]
        
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        
        # COCO format: [x, y, width, height]
        return [int(x_min), int(y_min), int(x_max - x_min + 1), int(y_max - y_min + 1)]
    
    def process_image_and_mask(self, image_path: str, mask_path: str, 
                              class_mapping: Optional[Dict[int, Tuple[str, str]]] = None) -> bool:
        """
        Process a single image and its corresponding instance mask.
        
        Args:
            image_path: Path to the original image
            mask_path: Path to the instance mask (16-bit PNG)
            class_mapping: Optional mapping from instance_id to (class_name, supercategory)
                         If None, all instances will be assigned to "object" category
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load image to get dimensions
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Could not load image {image_path}")
                return False
                
            height, width = image.shape[:2]
            
            # Load instance mask
            instance_mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            if instance_mask is None:
                print(f"Warning: Could not load mask {mask_path}")
                return False
            
            # Get image filename
            image_filename = os.path.basename(image_path)
            
            # Add image info to COCO data
            image_info = {
                "id": self.image_id_counter,
                "width": width,
                "height": height,
                "file_name": image_filename,
                "license": 1,
                "date_captured": datetime.now().isoformat()
            }
            self.coco_data["images"].append(image_info)
            
            # Process each instance in the mask
            unique_instances = np.unique(instance_mask)
            
            for instance_id in unique_instances:
                if instance_id == 0:  # Skip background
                    continue
                    
                # Create binary mask for this instance
                binary_mask = (instance_mask == instance_id)
                
                # Skip if mask is too small
                if np.sum(binary_mask) < 10:  # Minimum 10 pixels
                    continue
                
                # Get category info
                if class_mapping and instance_id in class_mapping:
                    class_name, supercategory = class_mapping[instance_id]
                    category_id = instance_id
                else:
                    # Default category
                    class_name = "object"
                    supercategory = "thing"
                    category_id = 1
                
                # Add category if not exists
                self.add_category(category_id, class_name, supercategory)
                
                # Get bounding box
                bbox = self.get_bbox_from_mask(binary_mask)
                
                # Convert mask to RLE
                rle = self.mask_to_rle(binary_mask)
                
                # Calculate area
                area = int(np.sum(binary_mask))
                
                # Create annotation
                annotation = {
                    "id": self.annotation_id_counter,
                    "image_id": self.image_id_counter,
                    "category_id": self.category_id_map[category_id],
                    "segmentation": rle,
                    "area": area,
                    "bbox": bbox,
                    "iscrowd": 0
                }
                
                self.coco_data["annotations"].append(annotation)
                self.annotation_id_counter += 1
            
            self.image_id_counter += 1
            return True
            
        except Exception as e:
            print(f"Error processing {image_path} and {mask_path}: {e}")
            return False
    
    def convert_dataset(self, images_dir: str, masks_dir: str, 
                       class_mapping: Optional[Dict[int, Tuple[str, str]]] = None,
                       image_extensions: List[str] = ['.jpg', '.jpeg', '.png'],
                       mask_extensions: List[str] = ['.png', '.jpg', '.jpeg']) -> Dict:
        """
        Convert entire dataset of images and instance masks to COCO format.
        
        Args:
            images_dir: Directory containing original images
            masks_dir: Directory containing instance masks
            class_mapping: Optional mapping from instance_id to (class_name, supercategory)
            image_extensions: List of valid image file extensions
            mask_extensions: List of valid mask file extensions
        
        Returns:
            COCO format dictionary
        """
        print(f"Converting dataset to COCO format...")
        print(f"Images directory: {images_dir}")
        print(f"Masks directory: {masks_dir}")
        
        # Reset for new conversion
        self.reset_annotations()
        
        # Add default category if no class mapping provided
        if not class_mapping:
            self.add_category(1, "object", "thing")
        
        processed_count = 0
        skipped_count = 0
        
        # Find all image files (support .jpg, .jpeg, .png)
        image_files = []
        for file in os.listdir(images_dir):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(file)
        
        print(f"Found {len(image_files)} image files")
        
        # Process each image
        for image_file in image_files:
            # Construct paths
            image_path = os.path.join(images_dir, image_file)
            
            # Find corresponding mask file (try all mask extensions)
            base_name = os.path.splitext(image_file)[0]
            mask_path = None
            for mask_ext in mask_extensions:
                candidate_mask = os.path.join(masks_dir, base_name + mask_ext)
                if os.path.exists(candidate_mask):
                    mask_path = candidate_mask
                    break
            
            if not mask_path:
                print(f"Warning: No mask found for {image_file} (tried extensions: {mask_extensions})")
                skipped_count += 1
                continue
            
            # Process the image-mask pair
            success = self.process_image_and_mask(image_path, mask_path, class_mapping)
            
            if success:
                processed_count += 1
                print(f"Processed {processed_count}/{len(image_files)}: {image_file}", end='\r')
            else:
                skipped_count += 1
        
        print(f"\nConversion complete!")
        print(f"Successfully processed: {processed_count} images")
        print(f"Skipped: {skipped_count} images")
        print(f"Total annotations: {len(self.coco_data['annotations'])}")
        print(f"Categories: {len(self.coco_data['categories'])}")
        
        return self.coco_data
    
    def save_coco_json(self, output_path: str) -> bool:
        """
        Save COCO annotations to JSON file.
        
        Args:
            output_path: Path to save the JSON file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(self.coco_data, f, indent=2)
            
            print(f"COCO annotations saved to: {output_path}")
            return True
            
        except Exception as e:
            print(f"Error saving COCO JSON: {e}")
            return False
    
    def validate_coco_format(self) -> bool:
        """
        Validate that the generated COCO data follows the correct format.
        
        Returns:
            True if valid, False otherwise
        """
        try:
            # Check required fields
            required_fields = ["info", "licenses", "categories", "images", "annotations"]
            for field in required_fields:
                if field not in self.coco_data:
                    print(f"Missing required field: {field}")
                    return False
            
            # Check that we have at least one category
            if len(self.coco_data["categories"]) == 0:
                print("No categories defined")
                return False
            
            # Check that we have images
            if len(self.coco_data["images"]) == 0:
                print("No images in dataset")
                return False
            
            # Validate image IDs are unique
            image_ids = [img["id"] for img in self.coco_data["images"]]
            if len(set(image_ids)) != len(image_ids):
                print("Duplicate image IDs found")
                return False
            
            # Validate annotation IDs are unique
            annotation_ids = [ann["id"] for ann in self.coco_data["annotations"]]
            if len(set(annotation_ids)) != len(annotation_ids):
                print("Duplicate annotation IDs found")
                return False
            
            print("COCO format validation passed!")
            return True
            
        except Exception as e:
            print(f"Validation error: {e}")
            return False
    
    def get_dataset_statistics(self) -> Dict:
        """
        Get statistics about the converted dataset.
        
        Returns:
            Dictionary with dataset statistics
        """
        stats = {
            "total_images": len(self.coco_data["images"]),
            "total_annotations": len(self.coco_data["annotations"]),
            "total_categories": len(self.coco_data["categories"]),
            "categories": {},
            "avg_instances_per_image": 0,
            "min_instances_per_image": float('inf'),
            "max_instances_per_image": 0
        }
        
        # Count annotations per category
        for category in self.coco_data["categories"]:
            cat_id = category["id"]
            cat_name = category["name"]
            count = sum(1 for ann in self.coco_data["annotations"] if ann["category_id"] == cat_id)
            stats["categories"][cat_name] = count
        
        # Count instances per image
        if stats["total_images"] > 0:
            image_instance_counts = {}
            for ann in self.coco_data["annotations"]:
                img_id = ann["image_id"]
                image_instance_counts[img_id] = image_instance_counts.get(img_id, 0) + 1
            
            if image_instance_counts:
                counts = list(image_instance_counts.values())
                stats["avg_instances_per_image"] = sum(counts) / len(counts)
                stats["min_instances_per_image"] = min(counts)
                stats["max_instances_per_image"] = max(counts)
            else:
                stats["min_instances_per_image"] = 0
        
        return stats


def create_class_mapping_from_semantic_masks(semantic_masks_dir: str, 
                                           class_names: Dict[int, str]) -> Dict[int, Tuple[str, str]]:
    """
    Helper function to create class mapping from semantic masks.
    
    Args:
        semantic_masks_dir: Directory containing semantic masks
        class_names: Mapping from class_id to class_name
        
    Returns:
        Mapping from instance_id to (class_name, supercategory)
    """
    # This is a simplified version - in practice, you'd need to correlate
    # instance masks with semantic masks to determine class for each instance
    class_mapping = {}
    
    for class_id, class_name in class_names.items():
        if class_id > 0:  # Skip background
            class_mapping[class_id] = (class_name, "thing")
    
    return class_mapping


# Example usage function
def convert_gui_dataset_to_coco(project_folder: str, output_path: str, 
                               dataset_name: str = "gui_dataset") -> bool:
    """
    Convert a GUI project dataset to COCO format.
    
    Args:
        project_folder: Path to GUI project folder (contains images/ and masks/)
        output_path: Path to save COCO JSON file
        dataset_name: Name for the dataset
        
    Returns:
        True if successful, False otherwise
    """
    images_dir = os.path.join(project_folder, "images")
    masks_dir = os.path.join(project_folder, "masks")
    
    # Check if directories exist
    if not os.path.exists(images_dir):
        print(f"Images directory not found: {images_dir}")
        return False
    
    if not os.path.exists(masks_dir):
        print(f"Masks directory not found: {masks_dir}")
        return False
    
    # Create converter
    converter = COCOConverter(
        dataset_name=dataset_name,
        dataset_description=f"Dataset converted from GUI project: {project_folder}"
    )
    
    # Convert dataset
    coco_data = converter.convert_dataset(images_dir, masks_dir)
    
    # Validate format
    if not converter.validate_coco_format():
        print("COCO format validation failed")
        return False
    
    # Print statistics
    stats = converter.get_dataset_statistics()
    print("\nDataset Statistics:")
    print(f"Total images: {stats['total_images']}")
    print(f"Total annotations: {stats['total_annotations']}")
    print(f"Total categories: {stats['total_categories']}")
    print(f"Average instances per image: {stats['avg_instances_per_image']:.1f}")
    
    # Save to file
    success = converter.save_coco_json(output_path)
    
    return success


if __name__ == "__main__":
    # Example usage
    project_folder = "Materials/CCPS"  # Example project folder
    output_path = "annotations/instances_train.json"
    
    success = convert_gui_dataset_to_coco(project_folder, output_path)
    
    if success:
        print("Conversion successful!")
    else:
        print("Conversion failed!")