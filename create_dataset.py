#!/usr/bin/env python
"""
Create Balanced Flake Dataset

This script automatically creates a dataset of images and masks from favorited flakes.
It downloads a specified number of flakes for each given thickness of a given material,
ensuring a balanced dataset across all thickness classes.

Output structure:
    dataset/
    ├── images/
    │   ├── flake_001.png
    │   ├── flake_002.png
    │   └── ...
    ├── masks/
    │   ├── flake_001.png
    │   ├── flake_002.png
    │   └── ...
    └── metadata.json

The metadata.json contains information about each flake (thickness, scan, etc.)
for training/validation purposes.
"""

import gzip
import json
import shutil
import sys
from datetime import datetime
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile
import requests



def decompress_response(response):
    """Decompress gzip response if needed."""
    try:
        if response.headers.get("Content-Encoding") == "gzip":
            try:
                return json.loads(gzip.decompress(response.content))
            except gzip.BadGzipFile:
                pass
        return response.json()
    except json.JSONDecodeError:
        print(f"Error: Received non-JSON response from server:")
        print(f"  Status: {response.status_code}")
        print(f"  Content: {response.text[:200]}...")
        return []


def get_material_thickness_combinations(base_url: str) -> dict:
    """Fetch all material-thickness combinations from the API."""
    try:
        response = requests.get(f"{base_url}/stats/uniqueCombinations")
        response.raise_for_status()
        return decompress_response(response)
    except requests.RequestException as e:
        print(f"Error fetching combinations: {e}")
        return {}


def get_flakes_by_thickness(base_url: str, material: str, thickness: str) -> list:
    """Fetch all favorite flakes for a specific material and thickness."""
    try:
        params = {
            "chip_material": material,
            "flake_thickness": thickness
        }
        response = requests.get(f"{base_url}/flakes", params=params)
        response.raise_for_status()
        flakes = decompress_response(response)
        
        if not isinstance(flakes, list):
            return []
        
        # Filter only favorites
        favorites = [f for f in flakes if f.get("flake_favorite", False)]
        
        # Sort by scan time (most recent first)
        favorites = sorted(favorites, key=lambda f: f.get("scan_time", 0), reverse=True)
        
        return favorites
    except requests.RequestException as e:
        print(f"Error fetching flakes: {e}")
        return []


def download_flake(base_url: str, flake_id: int) -> bytes:
    """Download a single flake as a ZIP file."""
    try:
        params = {"flake_id": flake_id}
        response = requests.get(f"{base_url}/download/flake", params=params)
        response.raise_for_status()
        return response.content
    except requests.RequestException as e:
        print(f"Error downloading flake {flake_id}: {e}")
        return None


def extract_image_and_mask(zip_content: bytes, output_name: str, images_dir: Path, masks_dir: Path) -> bool:
    """Extract raw_img and flake_mask from ZIP and save with clean names."""
    try:
        with ZipFile(BytesIO(zip_content), 'r') as zip_file:
            raw_img_saved = False
            mask_saved = False
            
            for filename in zip_file.namelist():
                base_name = filename.rsplit(".", 1)[0] if "." in filename else filename
                extension = filename.rsplit(".", 1)[1] if "." in filename else "png"
                
                if "raw_img" in base_name.lower():
                    content = zip_file.read(filename)
                    output_path = images_dir / f"{output_name}.{extension}"
                    output_path.write_bytes(content)
                    raw_img_saved = True
                    
                elif "flake_mask" in base_name.lower():
                    content = zip_file.read(filename)
                    output_path = masks_dir / f"{output_name}.{extension}"
                    output_path.write_bytes(content)
                    mask_saved = True
            
            return raw_img_saved and mask_saved
    except Exception as e:
        print(f"Error extracting: {e}")
        return False


def create_dataset(
    url: str,
    material: str,
    flakes_per_thickness: int = 10,
    output_dir: str = "./dataset",
    thicknesses: list = None
):
    """
    Create a balanced dataset with images and masks.
    
    Args:
        url: Base URL of the API
        material: Material to download (e.g., "Graphene")
        flakes_per_thickness: Number of flakes to download per thickness class
        output_dir: Output directory for the dataset
        thicknesses: List of specific thicknesses to include (None = all available)
    """
    base_url = url.rstrip("/")
    output_path = Path(output_dir)
    
    print("=" * 70)
    print("  Create Balanced Flake Dataset")
    print("=" * 70)
    
    # Test connection
    print(f"\nConnecting to {base_url}...")
    try:
        requests.get(base_url, timeout=5)
        print("Connected successfully!")
    except requests.RequestException as e:
        print(f"Error: Cannot connect to {base_url}")
        print(f"Details: {e}")
        return
    
    # Get available thicknesses for the material
    print(f"\nFetching available thicknesses for '{material}'...")
    combinations = get_material_thickness_combinations(base_url)
    
    if material not in combinations:
        print(f"Error: Material '{material}' not found!")
        print(f"Available materials: {', '.join(combinations.keys())}")
        return
    
    available_thicknesses = combinations[material]
    
    # Filter to requested thicknesses if specified
    if thicknesses:
        available_thicknesses = [t for t in available_thicknesses if t in thicknesses]
        missing = [t for t in thicknesses if t not in combinations[material]]
        if missing:
            print(f"Warning: These thicknesses not found: {missing}")
    
    print(f"\nThicknesses to process: {len(available_thicknesses)}")
    for t in available_thicknesses:
        print(f"  - {t}")
    
    print(f"\nTarget: {flakes_per_thickness} flakes per thickness")
    print(f"Output: {output_path.absolute()}")
    
    # Create output directories
    images_dir = output_path / "images"
    masks_dir = output_path / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    
    # Track metadata and statistics
    metadata = {
        "material": material,
        "created": datetime.now().isoformat(),
        "flakes_per_thickness": flakes_per_thickness,
        "thicknesses": {},
        "flakes": []
    }
    
    stats = {
        "total_downloaded": 0,
        "total_skipped": 0,
        "by_thickness": {}
    }
    
    global_index = 1
    
    # Process each thickness
    print("\n" + "=" * 70)
    for thickness in available_thicknesses:
        print(f"\n[{thickness}]")
        print("-" * 40)
        
        # Get favorite flakes for this thickness
        flakes = get_flakes_by_thickness(base_url, material, thickness)
        favorites_count = len(flakes)
        
        print(f"  Found {favorites_count} favorite flakes")
        
        if favorites_count == 0:
            print(f"  Skipping - no favorites found")
            stats["by_thickness"][thickness] = {"available": 0, "downloaded": 0}
            continue
        
        # Limit to requested number
        flakes_to_download = flakes[:flakes_per_thickness]
        
        downloaded_count = 0
        
        for flake in flakes_to_download:
            flake_id = flake.get("flake_id")
            scan_name = flake.get("scan_name", "unknown")
            size = flake.get("flake_size", 0)
            
            # Create output filename with zero-padded index
            output_name = f"flake_{global_index:04d}"
            
            print(f"  Downloading flake {flake_id} -> {output_name}...", end=" ")
            
            # Download flake ZIP
            zip_content = download_flake(base_url, flake_id)
            
            if zip_content and extract_image_and_mask(zip_content, output_name, images_dir, masks_dir):
                print("OK")
                downloaded_count += 1
                stats["total_downloaded"] += 1
                
                # Add to metadata
                metadata["flakes"].append({
                    "filename": output_name,
                    "flake_id": flake_id,
                    "thickness": thickness,
                    "material": material,
                    "scan_name": scan_name,
                    "size_um2": size,
                    "scan_time": flake.get("scan_time", 0)
                })
                
                global_index += 1
            else:
                print("FAILED")
                stats["total_skipped"] += 1
        
        stats["by_thickness"][thickness] = {
            "available": favorites_count,
            "downloaded": downloaded_count
        }
        metadata["thicknesses"][thickness] = downloaded_count
        
        print(f"  Downloaded: {downloaded_count}/{min(flakes_per_thickness, favorites_count)}")
    
    # Save metadata
    metadata_path = output_path / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 70)
    print("  Dataset Creation Complete!")
    print("=" * 70)
    print(f"\n  Material: {material}")
    print(f"  Total flakes downloaded: {stats['total_downloaded']}")
    print(f"  Total skipped/failed: {stats['total_skipped']}")
    print(f"\n  Flakes per thickness:")
    for thickness, counts in stats["by_thickness"].items():
        print(f"    {thickness}: {counts['downloaded']}/{counts['available']} available")
    
    print(f"\n  Output structure:")
    print(f"    {output_path}/")
    print(f"    +-- images/     ({stats['total_downloaded']} files)")
    print(f"    +-- masks/      ({stats['total_downloaded']} files)")
    print(f"    +-- metadata.json")
    
    


if __name__ == "__main__":
    # ============================================================
    # CONFIGURATION
    # ============================================================
    
    # API URL (Backend, not Frontend!)
    URL = "http://134.61.8.242:4999"
    date = datetime.now().strftime("%Y-%m-%d")
    
    # Material to download
    MATERIAL = "WSe2"
    
    # Number of flakes to download PER THICKNESS
    # The script will try to get this many for each thickness class
    FLAKES_PER_THICKNESS = 10
    
    # Specific thicknesses to include (None = all available)
    # Example: ["1-Layer", "2-Layer", "3-Layer"]
    THICKNESSES = None
    
    # Output directory
    OUTPUT_DIR = "./WSe2"

    # ============================================================
    # RUN DATASET CREATION
    # ============================================================
    create_dataset(
        url=URL,
        material=MATERIAL,
        flakes_per_thickness=FLAKES_PER_THICKNESS,
        output_dir=OUTPUT_DIR,
        thicknesses=THICKNESSES
    )
