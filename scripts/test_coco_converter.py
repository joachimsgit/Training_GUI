#!/usr/bin/env python3
"""
Test script for COCO converter functionality.
Demonstrates how to convert instance masks to COCO format.
"""

import os
import sys
import json

# Add the scripts directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from coco_converter import COCOConverter, convert_gui_dataset_to_coco


def test_coco_converter():
    """Test the COCO converter with sample data."""
    
    print("=" * 60)
    print("COCO Converter Test")
    print("=" * 60)
    
    # Example: Convert WSe2 dataset if it exists
    project_folders = [
        "Materials/WSe2",
    ]
    
    for project_folder in project_folders:
        if os.path.exists(project_folder):
            print(f"\nTesting with project: {project_folder}")
            
            # Check if required directories exist
            images_dir = os.path.join(project_folder, "images")
            masks_dir = os.path.join(project_folder, "masks")
            
            if not os.path.exists(images_dir):
                print(f"  ❌ Images directory not found: {images_dir}")
                continue
                
            if not os.path.exists(masks_dir):
                print(f"  ❌ Masks directory not found: {masks_dir}")
                continue
            
            # Count files
            image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            mask_files = [f for f in os.listdir(masks_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            print(f"  📸 Found {len(image_files)} image files")
            print(f"  🎭 Found {len(mask_files)} mask files")
            
            if len(image_files) == 0 or len(mask_files) == 0:
                print("  ⚠️ Insufficient data for conversion")
                continue
            
            # Create output directory
            output_dir = os.path.join(project_folder, "coco_annotations")
            os.makedirs(output_dir, exist_ok=True)
            
            # Convert to COCO format
            output_path = os.path.join(output_dir, "instances_all.json")
            
            print(f"  🔄 Converting to COCO format...")
            success = convert_gui_dataset_to_coco(
                project_folder=project_folder,
                output_path=output_path,
                dataset_name=os.path.basename(project_folder)
            )
            
            if success:
                print(f"  ✅ Conversion successful!")
                print(f"  📄 COCO annotations saved to: {output_path}")
                
                # Load and display summary
                try:
                    with open(output_path, 'r') as f:
                        coco_data = json.load(f)
                    
                    print(f"  📊 Summary:")
                    print(f"     - Images: {len(coco_data['images'])}")
                    print(f"     - Annotations: {len(coco_data['annotations'])}")
                    print(f"     - Categories: {len(coco_data['categories'])}")
                    
                    # Show categories
                    if coco_data['categories']:
                        print(f"  🏷️ Categories:")
                        for cat in coco_data['categories']:
                            print(f"     - {cat['name']} (ID: {cat['id']})")
                            
                except Exception as e:
                    print(f"  ❌ Error reading generated file: {e}")
            else:
                print(f"  ❌ Conversion failed!")
            
            print("-" * 40)
            return  # Test with first available project
    
    print("\n⚠️ No suitable project folders found for testing.")
    print("Available projects should have 'images/' and 'masks/' subdirectories.")


def demo_advanced_usage():
    """Demonstrate advanced converter usage with custom class mapping."""
    
    print("\n" + "=" * 60)
    print("Advanced COCO Converter Demo")
    print("=" * 60)
    
    # Create a converter instance
    converter = COCOConverter(
        dataset_name="advanced_demo",
        dataset_description="Demo of advanced COCO converter features"
    )
    
    # Add custom categories
    converter.add_category(1, "background", "stuff")
    converter.add_category(2, "flake_type_1", "thing") 
    converter.add_category(3, "flake_type_2", "thing")
    converter.add_category(4, "contaminant", "thing")
    
    print("✅ Added custom categories:")
    for cat in converter.coco_data['categories']:
        print(f"   - {cat['name']} (ID: {cat['id']}, Super: {cat['supercategory']})")
    
    # Show validation
    print(f"\n🔍 Format validation: {'✅ PASS' if converter.validate_coco_format() else '❌ FAIL'}")
    
    print("\nAdvanced features available:")
    print("- Custom class mapping from instance IDs to categories")
    print("- RLE mask encoding for efficient storage")
    print("- Automatic bounding box generation")
    print("- Dataset statistics and validation")
    print("- Flexible file naming conventions")


if __name__ == "__main__":
    test_coco_converter()
    demo_advanced_usage()
    
    print("\n" + "=" * 60)
    print("COCO Converter Ready for Use! 🚀")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Use convert_gui_dataset_to_coco() for simple conversions")
    print("2. Use COCOConverter class directly for advanced control")
    print("3. Integrate into M2F training dialog")
    print("4. Generate train/val/test splits as needed")