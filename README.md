# Training GUI for MaskTerial

A PyQt5-based graphical user interface for training and using machine learning models for material flake detection and classification.

## Features

- **Image Annotation**: Mark regions of interest on microscopy images
- **GMM Training**: Train Gaussian Mixture Models for contrast-based classification
- **AMM Training**: Train Autoregressive Mixture Models for advanced classification
- **M2F Training**: Train Mask2Former segmentation models *(🚧 Work in Progress)*
- **Class Annotation**: Annotate instance classes using contrast data visualization
- **Inference GUI**: Run inference on new images using trained models

## Requirements

- **Python 3.10 or 3.11** (recommended for best compatibility)
- CUDA-compatible GPU (recommended for deep learning features)

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/joachimsgit/Training_GUI.git
cd Training_GUI
```

### 2. Create a virtual environment (recommended)

```bash
# Using venv
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on Linux/Mac
source venv/bin/activate
```

Or using conda:
```bash
conda create -n training_gui python=3.10
conda activate training_gui
```

### 3. Install PyTorch (with CUDA support if available)

Visit [PyTorch Get Started](https://pytorch.org/get-started/locally/) and select your configuration.

Example for CUDA 11.8:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

For CPU only:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 4. Install Detectron2

Follow the [Detectron2 installation guide](https://detectron2.readthedocs.io/en/latest/tutorials/install.html).

For Linux:
```bash
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

For Windows, you may need to build from source or use pre-built wheels.

### 5. Install remaining dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Training GUI (Main Application)

```bash
python training_GUI.py
```

This opens the main window for:
- Loading image datasets
- Annotating flake regions
- Training classification models (GMM, AMM)
- Training segmentation models (M2F) - *coming soon*

### Inference GUI

```bash
python inference_gui.py
```

A simplified interface for running inference on new images using pre-trained models.

## Project Structure

```
Training_GUI/
├── training_GUI.py          # Main entry point for training GUI
├── inference_gui.py         # Standalone inference GUI
├── help_texts.py            # Help documentation strings
├── gui/                     # GUI components
│   ├── main_window.py       # Main application window
│   ├── Image_viewer.py      # Image display widget
│   ├── plot_canvas.py       # Matplotlib plotting widget
│   ├── gmm_training_dialog.py
│   ├── amm_training_dialog.py
│   ├── m2f_training_dialog.py  # (Work in Progress)
│   ├── class_annotator_dialog.py
│   └── help_dialog.py
├── maskterial/              # Core ML library
│   ├── maskterial.py        # Main MaskTerial class
│   ├── modeling/            # Model implementations
│   │   ├── classification_models/
│   │   ├── segmentation_models/
│   │   └── postprocessing_models/
│   ├── structures/          # Data structures
│   └── utils/               # Utility functions
├── scripts/                 # Helper scripts
│   ├── preprocessor_functions.py
│   ├── fitting_functions.py
│   ├── plotting_functions.py
│   ├── mask_functions.py
│   └── coco_converter.py
└── Materials/               # Example datasets (not tracked in git)
```

## Data Organization

The GUI expects material datasets to be organized as follows:

```
Materials/
└── your_material/
    ├── images/              # Original microscopy images
    ├── masks/               # Instance segmentation masks
    ├── semantic_masks/      # (Optional) Semantic class masks
    ├── contrast_data/       # Generated contrast data files
    ├── GMM/                 # Trained GMM models
    └── AMM/                 # Trained AMM models
```

## Troubleshooting

### PyQt5 issues on Linux
```bash
sudo apt-get install python3-pyqt5
```

### CUDA out of memory
- Reduce batch size in training dialogs
- Use a smaller model configuration

### Detectron2 installation issues
- Ensure your PyTorch version matches the detectron2 requirements
- On Windows, consider using WSL2 for easier installation