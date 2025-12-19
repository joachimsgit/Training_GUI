#!/usr/bin/env python3
"""
Minimal Inference GUI for MaskTerial
Based on demo_inference.ipynb workflow

This is a simplified standalone GUI for loading models and running inference
on individual images.
"""

import os
import sys
import traceback
from typing import Optional, Any

import cv2
import numpy as np
# Lazy load torch - will be imported when needed
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QDoubleSpinBox, QSpinBox,
    QFileDialog, QMessageBox, QScrollArea, QTextEdit, QGroupBox,
    QGridLayout, QProgressBar
)

# Add parent directory to path for importing maskterial
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Use smart preloading - start loading heavy libraries in background when user interacts
MASKTERIAL_AVAILABLE = None  # Will be determined when first needed
MaskTerial = None
load_models = None
Flake = None
_torch_module = None  # Cache torch module when loaded

def lazy_load_libraries():
    """Lazy load heavy libraries only when needed"""
    global MASKTERIAL_AVAILABLE, MaskTerial, load_models, Flake, _torch_module
    
    if MASKTERIAL_AVAILABLE is None:
        print("Loading PyTorch and MaskTerial libraries (this may take ~15 seconds)...")
        try:
            # Load torch first
            import torch
            _torch_module = torch
            
            # Then load MaskTerial
            from maskterial import MaskTerial as _MaskTerial, load_models as _load_models
            from maskterial.structures import Flake as _Flake
            
            MaskTerial = _MaskTerial
            load_models = _load_models
            Flake = _Flake
            MASKTERIAL_AVAILABLE = True
            print("Libraries loaded successfully!")
            return True, torch
        except ImportError as e:
            print(f"Warning: Libraries not available: {e}")
            MASKTERIAL_AVAILABLE = False
            return False, None
    
    # If already loaded, return cached torch
    return MASKTERIAL_AVAILABLE, _torch_module


class BackgroundPreloader(QThread):
    """Thread that preloads heavy libraries in background when user starts interacting"""
    finished = pyqtSignal(bool, object)  # success, torch_module
    progress = pyqtSignal(str)
    
    def run(self):
        try:
            self.progress.emit("🔄 Preloading libraries in background...")
            success, torch_module = lazy_load_libraries()
            if success:
                self.progress.emit("✅ Libraries preloaded! Model loading will be instant.")
            else:
                self.progress.emit("❌ Failed to preload libraries.")
            self.finished.emit(success, torch_module)
        except Exception as e:
            self.progress.emit(f"❌ Preloading error: {e}")
            self.finished.emit(False, None)


class ModelLoadThread(QThread):
    """Thread for loading models to avoid GUI freezing"""
    finished = pyqtSignal(object, object, object)  # segmentation, classification, postprocessing models
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, seg_model_type, cls_model_type, pp_model_type, 
                 seg_model_root, cls_model_root, pp_model_root, device):
        super().__init__()
        self.seg_model_type = seg_model_type
        self.cls_model_type = cls_model_type
        self.pp_model_type = pp_model_type
        self.seg_model_root = seg_model_root
        self.cls_model_root = cls_model_root
        self.pp_model_root = pp_model_root
        self.device = device

    def run(self):
        try:
            # Lazy load libraries in the thread
            success, torch = lazy_load_libraries()
            if not success:
                self.error.emit("Required libraries are not available. Please install them first.")
                return

            self.progress.emit("Loading models...")
            
            segmentation_model, classification_model, postprocessing_model = load_models(
                seg_model_type=self.seg_model_type,
                seg_model_root=self.seg_model_root,
                cls_model_type=self.cls_model_type,
                cls_model_root=self.cls_model_root,
                pp_model_type=self.pp_model_type,
                pp_model_root=self.pp_model_root,
                device=self.device,
            )
            
            self.progress.emit("Models loaded successfully!")
            self.finished.emit(segmentation_model, classification_model, postprocessing_model)
            
        except Exception as e:
            error_msg = f"Failed to load models: {str(e)}\n\n{traceback.format_exc()}"
            self.error.emit(error_msg)


class InferenceThread(QThread):
    """Thread for running inference to avoid GUI freezing"""
    finished = pyqtSignal(np.ndarray, list)  # processed image, flakes list
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, predictor, image):
        super().__init__()
        self.predictor = predictor
        self.image = image

    def run(self):
        try:
            self.progress.emit("Running inference...")
            
            flakes = self.predictor.predict(self.image)
            
            # Create a copy of the image for visualization
            result_image = self.image.copy()
            
            # Apply visualization similar to demo notebook
            colors = [
                (255, 0, 0),      # Red
                (0, 0, 255),      # Blue
                (0, 255, 0),      # Green
                (0, 255, 255),    # Cyan
                (255, 0, 255),    # Magenta
                (255, 41, 255),   # Pink
            ]
            
            for flake in flakes:
                mask = flake.mask.astype(np.uint8)
                class_id = int(flake.thickness)
                
                if class_id < len(colors):
                    color = colors[class_id]
                else:
                    color = (128, 128, 128)  # Gray for unknown classes

                # Draw outline
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(result_image, contours, -1, color, 2)

                # Get bounding box
                x, y, w, h = cv2.boundingRect(mask)

                # Draw bounding box
                cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 2)

                # Add class label
                label = f"Class {class_id}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 2

                # Get text size for background
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, font, font_scale, thickness
                )

                # Adjust text position to keep it within bounds
                text_y = y - 5 if y - text_height - 10 >= 0 else y + h + text_height + 5
                bg_y1 = text_y - text_height - 5 if y - text_height - 10 >= 0 else y + h
                bg_y2 = text_y + 5 if y - text_height - 10 >= 0 else y + h + text_height + 10

                # Draw background rectangle for text
                cv2.rectangle(result_image, (x, bg_y1), (x + text_width, bg_y2), color, -1)

                # Draw text
                cv2.putText(
                    result_image, label, (x, text_y), font, font_scale, (255, 255, 255), thickness
                )
            
            self.progress.emit(f"Inference complete! Detected {len(flakes)} flake(s).")
            self.finished.emit(result_image, flakes)
            
        except Exception as e:
            error_msg = f"Failed to run inference: {str(e)}\n\n{traceback.format_exc()}"
            self.error.emit(error_msg)


class MinimalInferenceGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Minimal MaskTerial Inference")
        self.setGeometry(100, 100, 1200, 800)
        
        # Model and inference state
        self.predictor: Optional[Any] = None  # Will be MaskTerial when loaded
        self.current_image: Optional[np.ndarray] = None
        self.device = "auto"  # Will be determined when torch is loaded
        
        # Model paths
        self.seg_model_path: Optional[str] = None
        self.cls_model_path: Optional[str] = None
        self.pp_model_path: Optional[str] = None
        
        # Flatfield correction
        self.flatfield_image: Optional[np.ndarray] = None
        self.flatfield_path: Optional[str] = None
        
        # Threads
        self.model_load_thread: Optional[ModelLoadThread] = None
        self.inference_thread: Optional[InferenceThread] = None
        self.preloader_thread: Optional[BackgroundPreloader] = None
        
        # Preloading state
        self.libraries_preloaded = False
        self.preloading_started = False
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel for controls
        left_panel = QWidget()
        left_panel.setMaximumWidth(350)
        left_layout = QVBoxLayout(left_panel)
        
        # Model configuration group
        model_group = QGroupBox("Model Configuration")
        model_layout = QGridLayout(model_group)
        
        # Segmentation model
        model_layout.addWidget(QLabel("Segmentation:"), 0, 0)
        self.seg_model_combo = QComboBox()
        self.seg_model_combo.addItems(["M2F", "None"])
        self.seg_model_combo.currentTextChanged.connect(self.on_model_type_changed)
        model_layout.addWidget(self.seg_model_combo, 0, 1)
        
        # Classification model
        model_layout.addWidget(QLabel("Classification:"), 1, 0)
        self.cls_model_combo = QComboBox()
        self.cls_model_combo.addItems(["AMM", "GMM", "None"])
        self.cls_model_combo.currentTextChanged.connect(self.on_model_type_changed)
        model_layout.addWidget(self.cls_model_combo, 1, 1)
        
        # Post-processing model
        model_layout.addWidget(QLabel("Post-processing:"), 2, 0)
        self.pp_model_combo = QComboBox()
        self.pp_model_combo.addItems(["None", "L2"])
        self.pp_model_combo.currentTextChanged.connect(self.on_model_type_changed)
        model_layout.addWidget(self.pp_model_combo, 2, 1)
        
        # Segmentation model path
        model_layout.addWidget(QLabel("Segmentation Model:"), 3, 0)
        self.seg_model_btn = QPushButton("Browse Segmentation Model")
        self.seg_model_btn.clicked.connect(self.select_segmentation_model)
        model_layout.addWidget(self.seg_model_btn, 3, 1)
        
        self.seg_model_path_label = QLabel("No model selected")
        self.seg_model_path_label.setWordWrap(True)
        self.seg_model_path_label.setStyleSheet("color: gray; font-size: 10px;")
        model_layout.addWidget(self.seg_model_path_label, 4, 0, 1, 2)
        
        # Classification model path
        model_layout.addWidget(QLabel("Classification Model:"), 5, 0)
        self.cls_model_btn = QPushButton("Browse Classification Model")
        self.cls_model_btn.clicked.connect(self.select_classification_model)
        model_layout.addWidget(self.cls_model_btn, 5, 1)
        
        self.cls_model_path_label = QLabel("No model selected")
        self.cls_model_path_label.setWordWrap(True)
        self.cls_model_path_label.setStyleSheet("color: gray; font-size: 10px;")
        model_layout.addWidget(self.cls_model_path_label, 6, 0, 1, 2)
        
        # Post-processing model path
        model_layout.addWidget(QLabel("Post-processing Model:"), 7, 0)
        self.pp_model_btn = QPushButton("Browse Post-processing Model")
        self.pp_model_btn.clicked.connect(self.select_postprocessing_model)
        model_layout.addWidget(self.pp_model_btn, 7, 1)
        
        self.pp_model_path_label = QLabel("No model selected")
        self.pp_model_path_label.setWordWrap(True)
        self.pp_model_path_label.setStyleSheet("color: gray; font-size: 10px;")
        model_layout.addWidget(self.pp_model_path_label, 8, 0, 1, 2)
        
        left_layout.addWidget(model_group)
        
        # Inference parameters group
        params_group = QGroupBox("Inference Parameters")
        params_layout = QGridLayout(params_group)
        
        # Score threshold
        params_layout.addWidget(QLabel("Score Threshold:"), 0, 0)
        self.score_threshold_spin = QDoubleSpinBox()
        self.score_threshold_spin.setRange(0.0, 1.0)
        self.score_threshold_spin.setSingleStep(0.1)
        self.score_threshold_spin.setValue(0.1)
        self.score_threshold_spin.setDecimals(2)
        self.score_threshold_spin.valueChanged.connect(lambda: self.start_background_preloading())
        params_layout.addWidget(self.score_threshold_spin, 0, 1)
        
        # Min class occupancy
        params_layout.addWidget(QLabel("Min Class Occupancy:"), 1, 0)
        self.min_class_occupancy_spin = QDoubleSpinBox()
        self.min_class_occupancy_spin.setRange(0.0, 1.0)
        self.min_class_occupancy_spin.setSingleStep(0.1)
        self.min_class_occupancy_spin.setValue(0.5)
        self.min_class_occupancy_spin.setDecimals(2)
        params_layout.addWidget(self.min_class_occupancy_spin, 1, 1)
        
        # Size threshold
        params_layout.addWidget(QLabel("Size Threshold:"), 2, 0)
        self.size_threshold_spin = QSpinBox()
        self.size_threshold_spin.setRange(0, 10000)
        self.size_threshold_spin.setValue(200)
        params_layout.addWidget(self.size_threshold_spin, 2, 1)
        
        left_layout.addWidget(params_group)
        
        # Flatfield correction group
        flatfield_group = QGroupBox("Flatfield Correction")
        flatfield_layout = QGridLayout(flatfield_group)
        
        # Enable flatfield checkbox
        self.use_flatfield_checkbox = QComboBox()
        self.use_flatfield_checkbox.addItems(["Disabled", "Enabled"])
        self.use_flatfield_checkbox.currentTextChanged.connect(self.on_flatfield_toggle)
        flatfield_layout.addWidget(QLabel("Flatfield:"), 0, 0)
        flatfield_layout.addWidget(self.use_flatfield_checkbox, 0, 1)
        
        # Browse flatfield button
        self.flatfield_btn = QPushButton("Browse Flatfield Image")
        self.flatfield_btn.clicked.connect(self.select_flatfield_image)
        self.flatfield_btn.setEnabled(False)
        flatfield_layout.addWidget(self.flatfield_btn, 1, 0, 1, 2)
        
        # Flatfield path label
        self.flatfield_path_label = QLabel("No flatfield image selected")
        self.flatfield_path_label.setWordWrap(True)
        self.flatfield_path_label.setStyleSheet("color: gray; font-size: 10px;")
        flatfield_layout.addWidget(self.flatfield_path_label, 2, 0, 1, 2)
        
        left_layout.addWidget(flatfield_group)
        
        # Action buttons
        self.load_models_btn = QPushButton("Load Models")
        self.load_models_btn.clicked.connect(self.load_models)
        left_layout.addWidget(self.load_models_btn)
        
        self.load_image_btn = QPushButton("Load Image")
        self.load_image_btn.clicked.connect(self.load_image)
        self.load_image_btn.setEnabled(False)
        left_layout.addWidget(self.load_image_btn)
        
        self.run_inference_btn = QPushButton("Run Inference")
        self.run_inference_btn.clicked.connect(self.run_inference)
        self.run_inference_btn.setEnabled(False)
        left_layout.addWidget(self.run_inference_btn)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        left_layout.addWidget(self.progress_bar)
        
        # Status text
        self.status_text = QTextEdit()
        self.status_text.setMaximumHeight(150)
        self.status_text.setReadOnly(True)
        left_layout.addWidget(self.status_text)
        
        # Device info
        self.device_label = QLabel("Device: Will be detected when loading models")
        self.device_label.setAlignment(Qt.AlignCenter)
        font = QFont()
        font.setBold(True)
        self.device_label.setFont(font)
        left_layout.addWidget(self.device_label)
        
        left_layout.addStretch()
        
        # Right panel for image display
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Image display
        self.image_scroll = QScrollArea()
        self.image_label = QLabel("Load an image to begin inference")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 1px solid gray; background-color: white;")
        self.image_label.setMinimumSize(600, 400)
        
        self.image_scroll.setWidget(self.image_label)
        self.image_scroll.setWidgetResizable(True)
        right_layout.addWidget(self.image_scroll)
        
        # Add panels to main layout
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel, 1)  # Give more space to image panel
        
        # Initial status
        self.log_message("Ready! GUI loaded quickly using smart preloading.")
        self.log_message("Tip: Start selecting models - heavy libraries will preload in background!")
        self.log_message("Note: AMM models need model.pth, GMM models need GMM_parameters.json")
        
        # Update button states based on initial selections
        self.on_model_type_changed()
        
    def on_flatfield_toggle(self):
        """Handle enable/disable of flatfield correction"""
        enabled = self.use_flatfield_checkbox.currentText() == "Enabled"
        self.flatfield_btn.setEnabled(enabled)
        
        if not enabled:
            self.flatfield_path_label.setText("Flatfield correction disabled")
            self.flatfield_path_label.setStyleSheet("color: gray; font-size: 10px;")
        else:
            if self.flatfield_path:
                self.flatfield_path_label.setText(f"✓ {self.flatfield_path}")
                self.flatfield_path_label.setStyleSheet("color: green; font-size: 10px;")
            else:
                self.flatfield_path_label.setText("No flatfield image selected")
                self.flatfield_path_label.setStyleSheet("color: gray; font-size: 10px;")
    
    def select_flatfield_image(self):
        """Select a flatfield image for vignette correction"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Flatfield Image", "", 
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff *.tif)"
        )
        
        if file_path:
            try:
                # Load flatfield image
                flatfield = cv2.imread(file_path)
                if flatfield is None:
                    raise ValueError("Could not load the selected flatfield image")
                
                self.flatfield_image = flatfield
                self.flatfield_path = file_path
                
                self.flatfield_path_label.setText(f"✓ {file_path}")
                self.flatfield_path_label.setStyleSheet("color: green; font-size: 10px;")
                self.log_message(f"Flatfield image loaded: {os.path.basename(file_path)}")
                
            except Exception as e:
                self.log_message(f"Failed to load flatfield image: {e}")
                QMessageBox.critical(self, "Flatfield Loading Error", str(e))
    
    def remove_vignette(self, image: np.ndarray, flatfield: np.ndarray, 
                       max_background_value: int = 241) -> np.ndarray:
        """
        Removes the vignette from the image using flatfield correction
        
        Args:
            image: The image with vignette (NxMx3 array)
            flatfield: The flatfield in RGB (NxMx3 array)
            max_background_value: The maximum value of the background
            
        Returns:
            The image without vignette
        """
        image_no_vignette = image / flatfield * cv2.mean(flatfield)[:-1]
        image_no_vignette[image_no_vignette > max_background_value] = max_background_value
        return np.asarray(image_no_vignette, dtype=np.uint8)
    
    def start_background_preloading(self):
        """Start preloading heavy libraries in background when user starts interacting"""
        if not self.preloading_started and not self.libraries_preloaded:
            self.preloading_started = True
            self.log_message("🚀 Starting background preloading...")
            
            # Update button to show preloading is happening
            self.load_models_btn.setText("🔄 Load Models (Preloading...)")
            
            self.preloader_thread = BackgroundPreloader()
            self.preloader_thread.finished.connect(self.on_preloading_finished)
            self.preloader_thread.progress.connect(self.log_message)
            self.preloader_thread.start()
    
    def on_preloading_finished(self, success: bool, torch_module):
        """Handle completion of background preloading"""
        self.libraries_preloaded = success
        if success and torch_module:
            # Update device info now that torch is loaded
            if self.device == "auto":
                self.device = torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")
                self.device_label.setText(f"Device: {self.device} (preloaded)")
            
            # Update load models button to show preloading is complete
            self.load_models_btn.setText("🚀 Load Models (Ready!)")
        else:
            self.load_models_btn.setText("⚠️ Load Models (Libraries failed)")
        
    def select_segmentation_model(self):
        """Select the segmentation model directory"""
        # Trigger background preloading when user starts browsing
        self.start_background_preloading()
        
        directory = QFileDialog.getExistingDirectory(
            self, "Select Segmentation Model Directory",
            "" if not self.seg_model_path else self.seg_model_path
        )
        if directory:
            # Check for required files
            config_file = os.path.join(directory, "config.yaml")
            model_file = os.path.join(directory, "model_final.pth")
            
            if os.path.exists(config_file) and os.path.exists(model_file):
                self.seg_model_path = directory
                self.seg_model_path_label.setText(f"✓ {directory}")
                self.seg_model_path_label.setStyleSheet("color: green; font-size: 10px;")
                self.log_message(f"Segmentation model selected: {directory}")
            else:
                missing_files = []
                if not os.path.exists(config_file):
                    missing_files.append("config.yaml")
                if not os.path.exists(model_file):
                    missing_files.append("model_final.pth")
                
                QMessageBox.warning(
                    self, "Invalid Segmentation Model", 
                    f"Selected directory is missing required files:\n{', '.join(missing_files)}\n\n"
                    "A segmentation model directory should contain:\n• config.yaml\n• model_final.pth"
                )
                
    def select_classification_model(self):
        """Select the classification model directory"""
        # Trigger background preloading when user starts browsing
        self.start_background_preloading()
        
        directory = QFileDialog.getExistingDirectory(
            self, "Select Classification Model Directory",
            "" if not self.cls_model_path else self.cls_model_path
        )
        if directory:
            # Get the selected classification model type
            cls_model_type = self.cls_model_combo.currentText()
            
            # Check for required files based on model type
            if cls_model_type == "GMM":
                # GMM only requires GMM_parameters.json
                model_file = os.path.join(directory, "GMM_parameters.json")
                required_files = [model_file]
                model_type_name = "GMM"
                expected_files = "• GMM_parameters.json"
            else:  # AMM
                # AMM requires model.pth and associated files
                meta_file = os.path.join(directory, "meta_data.json")
                loc_file = os.path.join(directory, "loc.npy")
                cov_file = os.path.join(directory, "cov.npy")
                model_file = os.path.join(directory, "model.pth")
                required_files = [meta_file, loc_file, cov_file, model_file]
                model_type_name = "AMM"
                expected_files = "• meta_data.json\n• loc.npy\n• cov.npy\n• model.pth"
            
            missing_files = [os.path.basename(f) for f in required_files if not os.path.exists(f)]
            
            if not missing_files:
                self.cls_model_path = directory
                self.cls_model_path_label.setText(f"✓ {directory}")
                self.cls_model_path_label.setStyleSheet("color: green; font-size: 10px;")
                self.log_message(f"{model_type_name} classification model selected: {directory}")
            else:
                QMessageBox.warning(
                    self, "Invalid Classification Model", 
                    f"Selected directory is missing required files for {model_type_name}:\n{', '.join(missing_files)}\n\n"
                    f"A {model_type_name} classification model directory should contain:\n{expected_files}"
                )
    
    def validate_existing_classification_model(self):
        """Validate that the currently selected classification model matches the selected type"""
        if not self.cls_model_path:
            return
            
        cls_model_type = self.cls_model_combo.currentText()
        directory = self.cls_model_path
        
        # Check for required files based on current model type
        if cls_model_type == "GMM":
            # GMM only requires GMM_parameters.json
            model_file = os.path.join(directory, "GMM_parameters.json")
            required_files = [model_file]
            model_type_name = "GMM"
        else:  # AMM
            # AMM requires model.pth and associated files
            meta_file = os.path.join(directory, "meta_data.json")
            loc_file = os.path.join(directory, "loc.npy")
            cov_file = os.path.join(directory, "cov.npy")
            model_file = os.path.join(directory, "model.pth")
            required_files = [meta_file, loc_file, cov_file, model_file]
            model_type_name = "AMM"
        
        missing_files = [os.path.basename(f) for f in required_files if not os.path.exists(f)]
        
        if missing_files:
            # Current selection is not valid for the new model type
            self.cls_model_path_label.setText(f"⚠️ Invalid for {model_type_name} - please reselect")
            self.cls_model_path_label.setStyleSheet("color: orange; font-size: 10px;")
            self.cls_model_path = None
            self.log_message(f"Previously selected model is not compatible with {model_type_name}. Please select a new model.")
        else:
            # Current selection is still valid
            self.cls_model_path_label.setText(f"✓ {directory}")
            self.cls_model_path_label.setStyleSheet("color: green; font-size: 10px;")
            self.log_message(f"Current model is compatible with {model_type_name}.")
                
    def select_postprocessing_model(self):
        """Select the post-processing model directory"""
        # Trigger background preloading when user starts browsing
        self.start_background_preloading()
        
        directory = QFileDialog.getExistingDirectory(
            self, "Select Post-processing Model Directory",
            "" if not self.pp_model_path else self.pp_model_path
        )
        if directory:
            # Post-processing models may have different file requirements
            # For now, just accept any directory and let the loading handle validation
            self.pp_model_path = directory
            self.pp_model_path_label.setText(f"✓ {directory}")
            self.pp_model_path_label.setStyleSheet("color: green; font-size: 10px;")
            self.log_message(f"Post-processing model selected: {directory}")
            
    def on_model_type_changed(self):
        """Update UI based on selected model types"""
        # Trigger background preloading when user starts selecting models
        self.start_background_preloading()
        
        # Enable/disable browse buttons based on model selections
        seg_enabled = self.seg_model_combo.currentText() != "None"
        cls_enabled = self.cls_model_combo.currentText() != "None"
        pp_enabled = self.pp_model_combo.currentText() != "None"
        
        self.seg_model_btn.setEnabled(seg_enabled)
        self.cls_model_btn.setEnabled(cls_enabled)
        self.pp_model_btn.setEnabled(pp_enabled)
        
        # Update labels for disabled models
        if not seg_enabled:
            self.seg_model_path_label.setText("Segmentation disabled")
            self.seg_model_path_label.setStyleSheet("color: gray; font-size: 10px;")
            self.seg_model_path = None
        else:
            if not self.seg_model_path:
                self.seg_model_path_label.setText("No model selected")
                self.seg_model_path_label.setStyleSheet("color: gray; font-size: 10px;")
                
        if not cls_enabled:
            self.cls_model_path_label.setText("Classification disabled")
            self.cls_model_path_label.setStyleSheet("color: gray; font-size: 10px;")
            self.cls_model_path = None
        else:
            if not self.cls_model_path:
                self.cls_model_path_label.setText("No model selected")
                self.cls_model_path_label.setStyleSheet("color: gray; font-size: 10px;")
            else:
                # Re-validate existing selection if model type changed
                self.validate_existing_classification_model()
                
        if not pp_enabled:
            self.pp_model_path_label.setText("Post-processing disabled")
            self.pp_model_path_label.setStyleSheet("color: gray; font-size: 10px;")
            self.pp_model_path = None
        else:
            if not self.pp_model_path:
                self.pp_model_path_label.setText("No model selected")
                self.pp_model_path_label.setStyleSheet("color: gray; font-size: 10px;")
            
    def log_message(self, message: str):
        """Add a message to the status text"""
        self.status_text.append(message)
        # Auto-scroll to bottom
        cursor = self.status_text.textCursor()
        cursor.movePosition(cursor.End)
        self.status_text.setTextCursor(cursor)
        
    def load_models(self):
        """Load the selected models"""
        # Check if libraries are already preloaded
        if self.libraries_preloaded:
            self.log_message("✅ Using preloaded libraries - model loading will be fast!")
            success, torch = True, _torch_module
        else:
            self.log_message("⏳ Libraries not preloaded yet, loading now...")
            success, torch = lazy_load_libraries()
            
        if not success:
            QMessageBox.critical(self, "Error", "Required libraries are not available. Please install them first.")
            return
        
        # Update device info now that torch is loaded
        if self.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.device_label.setText(f"Device: {self.device}")
            
        # Validate model selections
        seg_model_type = self.seg_model_combo.currentText() if self.seg_model_combo.currentText() != "None" else None
        cls_model_type = self.cls_model_combo.currentText() if self.cls_model_combo.currentText() != "None" else None
        pp_model_type = self.pp_model_combo.currentText() if self.pp_model_combo.currentText() != "None" else None
        
        # Check if at least one model is selected
        if not seg_model_type and not cls_model_type:
            QMessageBox.warning(self, "No Models Selected", 
                              "Please select at least one model type (Segmentation or Classification).")
            return
        
        # Check if required models are selected
        if seg_model_type and not self.seg_model_path:
            QMessageBox.warning(self, "Missing Model", 
                              f"Please browse and select a {seg_model_type} segmentation model directory.")
            return
            
        if cls_model_type and not self.cls_model_path:
            QMessageBox.warning(self, "Missing Model", 
                              f"Please browse and select a {cls_model_type} classification model directory.")
            return
            
        if pp_model_type and not self.pp_model_path:
            QMessageBox.warning(self, "Missing Model", 
                              f"Please browse and select a {pp_model_type} post-processing model directory.")
            return
        
        # Use selected model paths
        seg_model_root = self.seg_model_path if seg_model_type else None
        cls_model_root = self.cls_model_path if cls_model_type else None
        pp_model_root = self.pp_model_path if pp_model_type else None
        
        # Disable UI during loading
        self.load_models_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        
        # Start loading thread
        self.model_load_thread = ModelLoadThread(
            seg_model_type, cls_model_type, pp_model_type,
            seg_model_root, cls_model_root, pp_model_root, self.device
        )
        self.model_load_thread.finished.connect(self.on_models_loaded)
        self.model_load_thread.error.connect(self.on_model_load_error)
        self.model_load_thread.progress.connect(self.log_message)
        self.model_load_thread.start()
        
    def on_models_loaded(self, segmentation_model, classification_model, postprocessing_model):
        """Handle successful model loading"""
        try:
            # Create predictor
            self.predictor = MaskTerial(
                segmentation_model=segmentation_model,
                classification_model=classification_model,
                postprocessing_model=postprocessing_model,
                score_threshold=self.score_threshold_spin.value(),
                min_class_occupancy=self.min_class_occupancy_spin.value(),
                size_threshold=self.size_threshold_spin.value(),
                device=self.device,
            )
            
            # Show summary of loaded models
            loaded_models = []
            if segmentation_model is not None:
                loaded_models.append(f"Segmentation ({self.seg_model_combo.currentText()})")
            if classification_model is not None:
                loaded_models.append(f"Classification ({self.cls_model_combo.currentText()})")
            if postprocessing_model is not None:
                loaded_models.append(f"Post-processing ({self.pp_model_combo.currentText()})")
            
            self.log_message(f"Models loaded successfully: {', '.join(loaded_models)}")
            self.log_message("You can now load an image for inference.")
            self.load_image_btn.setEnabled(True)
            
        except Exception as e:
            self.log_message(f"Error creating predictor: {e}")
        finally:
            self.load_models_btn.setEnabled(True)
            self.progress_bar.setVisible(False)
            
    def on_model_load_error(self, error_msg: str):
        """Handle model loading error"""
        self.log_message(f"Model loading failed: {error_msg}")
        QMessageBox.critical(self, "Model Loading Error", error_msg)
        self.load_models_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        
    def load_image(self):
        """Load an image for inference"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", 
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff *.tif)"
        )
        
        if file_path:
            try:
                # Load image with OpenCV
                self.current_image = cv2.imread(file_path)
                if self.current_image is None:
                    raise ValueError("Could not load the selected image")
                
                # Display image
                self.display_image(self.current_image)
                
                self.log_message(f"Image loaded: {os.path.basename(file_path)}")
                self.run_inference_btn.setEnabled(True)
                
            except Exception as e:
                self.log_message(f"Failed to load image: {e}")
                QMessageBox.critical(self, "Image Loading Error", str(e))
                
    def display_image(self, image: np.ndarray):
        """Display an image in the GUI"""
        try:
            # Convert BGR to RGB for Qt
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Convert to QImage
            height, width, channel = rgb_image.shape
            bytes_per_line = 3 * width
            q_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            
            # Convert to QPixmap and display
            pixmap = QPixmap.fromImage(q_image)
            
            # Scale image to fit display while maintaining aspect ratio
            scaled_pixmap = pixmap.scaled(
                800, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            
            self.image_label.setPixmap(scaled_pixmap)
            self.image_label.setAlignment(Qt.AlignCenter)
            
        except Exception as e:
            self.log_message(f"Error displaying image: {e}")
            
    def run_inference(self):
        """Run inference on the loaded image"""
        if self.predictor is None or self.current_image is None:
            return
        
        # Prepare the image for inference (apply flatfield correction if enabled)
        inference_image = self.current_image.copy()
        
        if self.use_flatfield_checkbox.currentText() == "Enabled":
            if self.flatfield_image is None:
                QMessageBox.warning(
                    self, "Flatfield Not Selected", 
                    "Flatfield correction is enabled but no flatfield image has been selected. "
                    "Please select a flatfield image or disable flatfield correction."
                )
                return
            
            # Check if flatfield and image have compatible dimensions
            if self.flatfield_image.shape != inference_image.shape:
                QMessageBox.warning(
                    self, "Dimension Mismatch", 
                    f"Flatfield image dimensions {self.flatfield_image.shape} do not match "
                    f"input image dimensions {inference_image.shape}. "
                    "Please use a flatfield image with matching dimensions."
                )
                return
            
            self.log_message("Applying flatfield correction...")
            inference_image = self.remove_vignette(inference_image, self.flatfield_image)
            self.log_message("Flatfield correction applied.")
            
        # Update predictor parameters
        self.predictor.score_threshold = self.score_threshold_spin.value()
        self.predictor.min_class_occupancy = self.min_class_occupancy_spin.value()
        self.predictor.size_threshold = self.size_threshold_spin.value()
        
        # Disable UI during inference
        self.run_inference_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        
        # Start inference thread with the (possibly corrected) image
        self.inference_thread = InferenceThread(self.predictor, inference_image)
        self.inference_thread.finished.connect(self.on_inference_finished)
        self.inference_thread.error.connect(self.on_inference_error)
        self.inference_thread.progress.connect(self.log_message)
        self.inference_thread.start()
        
    def on_inference_finished(self, result_image: np.ndarray, flakes: list):
        """Handle successful inference completion"""
        try:
            # Display result image with annotations
            self.display_image(result_image)
            
            # Log results
            if len(flakes) == 0:
                self.log_message("No flakes detected.")
            else:
                self.log_message(f"Detected {len(flakes)} flake(s):")
                for i, flake in enumerate(flakes):
                    self.log_message(f"  Flake {i+1}: Class {int(flake.thickness)}")
                    
        except Exception as e:
            self.log_message(f"Error processing results: {e}")
        finally:
            self.run_inference_btn.setEnabled(True)
            self.progress_bar.setVisible(False)
            
    def on_inference_error(self, error_msg: str):
        """Handle inference error"""
        self.log_message(f"Inference failed: {error_msg}")
        QMessageBox.critical(self, "Inference Error", error_msg)
        self.run_inference_btn.setEnabled(True)
        self.progress_bar.setVisible(False)


def main():
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("Minimal MaskTerial Inference")
    app.setApplicationVersion("1.0")
    
    window = MinimalInferenceGUI()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()