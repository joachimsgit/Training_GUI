import sys
import os
import cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog,
                             QGroupBox, QMessageBox)
from PyQt5.QtCore import Qt
import numpy as np
import matplotlib.cm as cm
from scipy.ndimage import gaussian_filter
from gui.Image_viewer import ImageViewer
from gui.plot_canvas import PlotCanvas
from gui.help_dialog import show_help_dialog

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.image_folder = ""
        self.image_files = []
        self.current_index = -1
        self.flatfield_path = None  # Path to flatfield image for vignette correction
        
        # Performance optimization: Image cache
        self.image_cache = {}  # Cache for loaded images {image_path: (pixmap, corrected_pixmap)}
        self.max_cache_size = 5  # Number of images to keep in cache
        
        # Initialize cumulative contrast data storage (note: contrast can be negative)
        self.cumulative_r_values = np.array([], dtype=np.float32)
        self.cumulative_g_values = np.array([], dtype=np.float32)
        self.cumulative_b_values = np.array([], dtype=np.float32)
        
        # Store mask information
        self.mask_files = {}  # Dictionary mapping image paths to mask paths
        self.semantic_mask_files = {}  # Dictionary mapping image paths to semantic mask paths
        self.showing_mask = False  # Flag to track if we're showing mask or original image
        self.showing_semantic_mask = False  # Flag to track if we're showing semantic mask
        
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("Image Viewer with Plots")
        self.setGeometry(100, 100, 1400, 900)
        self.showMaximized()  # Start maximized
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create header with title and help button
        header_layout = QHBoxLayout()
        
        # Application title
        title_label = QLabel("Training GUI")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #2c3e50; margin: 5px;")
        header_layout.addWidget(title_label)
        
        header_layout.addStretch()
        
        # Help button
        self.btn_help = QPushButton("❓")
        self.btn_help.setFixedSize(40, 30)
        self.btn_help.setToolTip("Show Help (F1)")
        self.btn_help.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 15px;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #21618c;
            }
        """)
        self.btn_help.clicked.connect(self.show_help)
        header_layout.addWidget(self.btn_help)
        
        main_layout.addLayout(header_layout)
        
        # Create horizontal layout for image and plots
        content_layout = QHBoxLayout()
        
        # Left side - Image viewer
        self.image_viewer = ImageViewer()
        self.image_viewer.setText("No Image Loaded")
        self.image_viewer.setMinimumSize(600, 400)
        content_layout.addWidget(self.image_viewer, stretch=2)
        
        # Right side - Plots
        plots_layout = QVBoxLayout()
        plots_group = QGroupBox("Contrast Plots")
        plots_group.setLayout(plots_layout)
        
        # Create three plots
        self.plot1 = PlotCanvas(self, "Red vs Green Contrast")
        self.plot2 = PlotCanvas(self, "Green vs Blue Contrast")
        self.plot3 = PlotCanvas(self, "Blue vs Red Contrast")

        plots_layout.addWidget(self.plot1)
        plots_layout.addWidget(self.plot2)
        plots_layout.addWidget(self.plot3)
        
        # Set fixed width for plots panel
        plots_group.setMaximumWidth(400)
        plots_group.setMinimumWidth(300)
        content_layout.addWidget(plots_group, stretch=1)
        
        # Add content layout to main layout
        main_layout.addLayout(content_layout)
        
        # Bottom - Control buttons
        controls_layout = QHBoxLayout()
        controls_group = QGroupBox("Controls")
        controls_group.setLayout(controls_layout)
        controls_group.setMaximumHeight(100)
        
        # Create buttons
        self.btn_select_folder = QPushButton("Select Project Folder")
        self.btn_previous = QPushButton("Previous Image (A)")
        self.btn_next = QPushButton("Next Image (D)")
        self.btn_save_mask = QPushButton("Save Mask (S)")
        self.btn_clear_markers = QPushButton("Clear Markers (C)")
        self.btn_toggle_mask = QPushButton("Show Mask (M)")
        self.btn_toggle_mask.setCheckable(True)
        self.btn_toggle_semantic = QPushButton("Show Semantic Mask (N)")
        self.btn_toggle_semantic.setCheckable(True)
        self.btn_toggle_plots = QPushButton("Show Single Image")
        self.btn_toggle_plots.setCheckable(True)
        self.btn_class_annotator = QPushButton("Annotate Classes")
        self.btn_train_amm = QPushButton("Train AMM Model")
        self.btn_train_gmm = QPushButton("Train GMM Model")
        self.btn_train_m2f = QPushButton("Train M2F Model (WIP)")
        self.btn_train_m2f.setToolTip("Work in Progress - Not yet available")
        
        # Initially disable navigation buttons
        self.btn_previous.setEnabled(False)
        self.btn_next.setEnabled(False)
        self.btn_toggle_mask.setEnabled(False)
        self.btn_toggle_semantic.setEnabled(False)
        self.btn_toggle_plots.setEnabled(False)
        self.btn_class_annotator.setEnabled(False)
        self.btn_train_amm.setEnabled(False)
        self.btn_train_gmm.setEnabled(False)
        self.btn_train_m2f.setEnabled(False)
        
        # Connect button signals
        self.btn_select_folder.clicked.connect(self.select_folder)
        self.btn_previous.clicked.connect(self.previous_image)
        self.btn_next.clicked.connect(self.next_image)
        self.btn_save_mask.clicked.connect(self.save_mask)
        self.btn_clear_markers.clicked.connect(self.clear_markers)
        self.btn_toggle_mask.clicked.connect(self.toggle_mask_view)
        self.btn_toggle_semantic.clicked.connect(self.toggle_semantic_mask_view)
        self.btn_toggle_plots.clicked.connect(self.toggle_plot_mode)
        self.btn_class_annotator.clicked.connect(self.open_class_annotator)
        self.btn_train_amm.clicked.connect(self.open_amm_training)
        self.btn_train_gmm.clicked.connect(self.open_gmm_training)
        self.btn_train_m2f.clicked.connect(self.open_m2f_training)
        
        # Add buttons to layout
        controls_layout.addWidget(self.btn_select_folder)
        controls_layout.addWidget(self.btn_previous)
        controls_layout.addWidget(self.btn_next)
        controls_layout.addWidget(self.btn_save_mask)
        controls_layout.addWidget(self.btn_clear_markers)
        controls_layout.addWidget(self.btn_toggle_mask)
        controls_layout.addWidget(self.btn_toggle_semantic)
        controls_layout.addWidget(self.btn_toggle_plots)
        controls_layout.addWidget(self.btn_class_annotator)
        controls_layout.addWidget(self.btn_train_amm)
        controls_layout.addWidget(self.btn_train_gmm)
        controls_layout.addWidget(self.btn_train_m2f)
        
        # Add status label
        self.status_label = QLabel("Ready")
        controls_layout.addWidget(self.status_label)
        controls_layout.addStretch()
        
        main_layout.addWidget(controls_group)
    
    def keyPressEvent(self, event):
        """Handle keyboard shortcuts"""
        key = event.key()
        
        # F1 - Show help
        if key == Qt.Key_F1:
            self.show_help()
        
        # S - Save mask
        elif key == Qt.Key_S:
            self.save_mask()
        
        # A - Previous picture
        elif key == Qt.Key_A:
            self.previous_image()
        
        # D - Next picture
        elif key == Qt.Key_D:
            self.next_image()
        
        # C - Clear markers
        elif key == Qt.Key_C:
            self.clear_markers()
        
        # M - Toggle mask view
        elif key == Qt.Key_M:
            if hasattr(self, 'btn_toggle_mask'):
                self.btn_toggle_mask.setChecked(not self.btn_toggle_mask.isChecked())
                self.toggle_mask_view()
        
        else:
            # Call parent implementation for unhandled keys
            super().keyPressEvent(event)
    
    def select_folder(self):
        """
        Open dialog to select project/material folder
        """
        project_folder = QFileDialog.getExistingDirectory(self, "Select Project/Material Folder")
        if project_folder:
            # Look for images folder within the project folder
            self.image_folder = ""
            for item in os.listdir(project_folder):
                item_path = os.path.join(project_folder, item)
                if os.path.isdir(item_path):
                    # Check if this folder contains images
                    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']
                    has_images = any(
                        any(file.lower().endswith(ext) for ext in image_extensions)
                        for file in os.listdir(item_path)
                        if os.path.isfile(os.path.join(item_path, file))
                    )
                    if has_images:
                        self.image_folder = item_path
                        break
            
            if not self.image_folder:
                QMessageBox.warning(self, "Warning", "No folder with images found in the selected project folder")
                return
            
            # Load images first (but don't load current image yet)
            self.load_images_list()
            
            # Find existing masks before loading the first image
            self.find_existing_masks(project_folder)
            self.find_existing_semantic_masks(project_folder)
            
            # Check for flatfield.png for vignette correction
            self.flatfield_path = None
            potential_flatfield = os.path.join(project_folder, "flatfield.png")
            if os.path.exists(potential_flatfield):
                self.flatfield_path = potential_flatfield
            else:
                # Also check in Flatfield folder
                flatfield_folder = os.path.join(project_folder, "Flatfield")
                if os.path.exists(flatfield_folder):
                    for file in os.listdir(flatfield_folder):
                        if file.lower().endswith('.png'):
                            self.flatfield_path = os.path.join(flatfield_folder, file)
                            break
            
            # Now load the first image with mask information available
            if self.image_files:
                self.current_index = 0
                self.load_current_image()
                self.btn_previous.setEnabled(True)
                self.btn_next.setEnabled(True)
                self.btn_toggle_plots.setEnabled(True)
                self.btn_class_annotator.setEnabled(True)
                self.btn_train_amm.setEnabled(True)
                self.btn_train_gmm.setEnabled(True)
                # M2F training is still work in progress - keep disabled
                # self.btn_train_m2f.setEnabled(True)
                self.status_label.setText(f"Loaded {len(self.image_files)} images")
            
            # Build cumulative data from individual contrast files
            self.load_cumulative_contrast_data(project_folder)
            
            # Update plots based on current mode
            if len(self.cumulative_r_values) > 0:
                if self.btn_toggle_plots.isChecked():
                    # Show single image data  
                    self.update_single_image_plots()
                else:
                    # Show cumulative data
                    self.update_rgb_contrast_plots()
            else:
                self.clear_plots()
    
    def load_images_list(self):
        """
        Load list of images from selected folder (without loading first image)
        """
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']
        self.image_files = []
        
        for file in os.listdir(self.image_folder):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                self.image_files.append(os.path.join(self.image_folder, file))
        
        self.image_files.sort()
        
        if not self.image_files:
            self.status_label.setText("No images found in folder")
            self.btn_previous.setEnabled(False)
            self.btn_next.setEnabled(False)
    
    def load_current_image(self):
        """Load and display current image with caching for better performance"""
        if 0 <= self.current_index < len(self.image_files):
            image_path = self.image_files[self.current_index]
            
            # Clear markers when switching images
            self.image_viewer.clear_markers()
            
            # Enable/disable mask toggle button based on mask existence
            has_mask = image_path in self.mask_files
            self.btn_toggle_mask.setEnabled(has_mask)
            self.btn_toggle_mask.setChecked(False)
            self.showing_mask = False
            
            # Enable/disable semantic mask toggle button
            has_semantic_mask = image_path in self.semantic_mask_files
            self.btn_toggle_semantic.setEnabled(has_semantic_mask)
            self.btn_toggle_semantic.setChecked(False)
            self.showing_semantic_mask = False
            
            # Try to use cached image first
            if self.use_cached_image(image_path):
                success = True
            else:
                success = self.image_viewer.set_image(image_path, self.flatfield_path)
                if success:
                    # Cache the loaded image
                    self.cache_current_image(image_path)
            
            if success:
                filename = os.path.basename(image_path)
                status = f"Image {self.current_index + 1}/{len(self.image_files)}: {filename}"
                if has_mask:
                    status += " (Mask available)"
                if self.flatfield_path:
                    status += " (Vignette corrected)"
                self.status_label.setText(status)
                
                # Defer plot updates to improve responsiveness
                self.update_plots_deferred()
    
    def use_cached_image(self, image_path):
        """Try to use a cached image for faster loading"""
        if image_path in self.image_cache:
            original_pixmap, corrected_pixmap = self.image_cache[image_path]
            
            # Set the appropriate pixmap based on whether we have flatfield correction
            if self.flatfield_path and corrected_pixmap:
                self.image_viewer.original_pixmap = corrected_pixmap
            else:
                self.image_viewer.original_pixmap = original_pixmap
                
            self.image_viewer.update_display()
            return True
        return False
        
    def cache_current_image(self, image_path):
        """Cache the currently loaded image"""
        if hasattr(self.image_viewer, 'original_pixmap') and self.image_viewer.original_pixmap:
            # Manage cache size
            if len(self.image_cache) >= self.max_cache_size:
                # Remove oldest entry (simple FIFO)
                oldest_key = next(iter(self.image_cache))
                del self.image_cache[oldest_key]
            
            # Store both original and vignette-corrected versions if available
            original_pixmap = self.image_viewer.original_pixmap
            corrected_pixmap = None
            
            # For now, store the current pixmap (which might be corrected)
            self.image_cache[image_path] = (original_pixmap, corrected_pixmap)
    
    def update_plots_deferred(self):
        """Update plots with a slight delay to improve UI responsiveness"""
        from PyQt5.QtCore import QTimer
        
        # Use a timer to defer plot updates
        QTimer.singleShot(50, self.update_plots_now)
    
    def update_plots_now(self):
        """Actually update the plots"""
        # Update plots based on current mode when navigating
        if self.btn_toggle_plots.isChecked():
            # Show single image data (always try to update, even if no cumulative data)
            self.update_single_image_plots()
        elif len(self.cumulative_r_values) > 0:
            # Show cumulative data (only if cumulative data exists)
            self.update_rgb_contrast_plots()
        else:
            # Clear plots if no data and in cumulative mode
            self.clear_plots()
    
    def previous_image(self):
        """
        Navigate to previous image with optimized loading
        """
        if self.current_index > 0:
            self.current_index -= 1
            self.load_current_image()
    
    def next_image(self):
        """
        Navigate to next image with optimized loading
        """
        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self.load_current_image()
    
    def save_mask(self):
        """Save the current instance mask"""
        if not self.image_files:
            return
            
        # Get the current image name
        current_image_path = self.image_files[self.current_index]
        current_image_name = os.path.basename(current_image_path)
        
        # Create mask name (same as image name)
        mask_name = current_image_name
        
        # Create masks directory if it doesn't exist
        mask_folder = os.path.join(os.path.dirname(self.image_folder), "masks")
        os.makedirs(mask_folder, exist_ok=True)
        
        # Create instance mask
        instance_mask = self.image_viewer.create_watershed_mask()
        
        if instance_mask is not None:
            # Save as 16-bit PNG to preserve instance IDs
            mask_path = os.path.join(mask_folder, mask_name)
            cv2.imwrite(mask_path, instance_mask.astype(np.uint16))
            
            # Update mask files dictionary
            self.mask_files[current_image_path] = mask_path
            
            # Extract RGB values from the masked regions and add to cumulative data
            success = self.extract_and_add_rgb_data(instance_mask)
            
            if success:
                self.status_label.setText(f"Mask saved as {mask_name}, cumulative data updated")
            else:
                self.status_label.setText(f"Mask saved as {mask_name} but no flake regions found")
            print(f"Mask saved successfully: {mask_path}")
        else:
            # Save empty mask (all zeros) when no markers are present
            if self.image_viewer.original_pixmap:
                # Get image dimensions from the original pixmap
                image_height = self.image_viewer.original_pixmap.height()
                image_width = self.image_viewer.original_pixmap.width()
            else:
                # Fallback to loading the image to get dimensions
                temp_image = cv2.imread(current_image_path)
                if temp_image is not None:
                    image_height, image_width = temp_image.shape[:2]
                else:
                    print("Error: Could not determine image dimensions for empty mask")
                    return
            
            # Create empty mask with same dimensions as image
            empty_mask = np.zeros((image_height, image_width), dtype=np.uint16)
            mask_path = os.path.join(mask_folder, mask_name)
            cv2.imwrite(mask_path, empty_mask)
            
            # Update mask files dictionary
            self.mask_files[current_image_path] = mask_path
            
            self.status_label.setText(f"Empty mask saved as {mask_name}")
            print(f"Empty mask saved: {mask_path}")

        # Enable toggle mask button now that mask exists
        self.btn_toggle_mask.setEnabled(True)
        
    def extract_and_add_rgb_data(self, instance_mask):
        """Extract RGB data from masked regions and add to cumulative data"""
        # Get the original image in BGR format for RGB analysis
        original_image = self.image_viewer.get_original_image_for_analysis()
        
        if original_image is None:
            print("Warning: Could not get original image for contrast analysis")
            return
            
        # Extract RGB values from the masked regions and add to cumulative data
        b, g, r = cv2.split(original_image)
        flake_pixels = instance_mask != 0
        
        project_folder = os.path.dirname(self.image_folder)    

        if np.any(flake_pixels):
            # Extract BGR values for flake regions (OpenCV uses BGR format)
            flake_b_values = b[flake_pixels]
            flake_g_values = g[flake_pixels]
            flake_r_values = r[flake_pixels]
            
            # Calculate background values (mean of non-flake regions)
            background_pixels = instance_mask == 0
            if np.any(background_pixels):
                bg_b_mean = np.mean(b[background_pixels])
                bg_g_mean = np.mean(g[background_pixels])
                bg_r_mean = np.mean(r[background_pixels])
                
                # Calculate contrast: (flake_color / background_color) - 1
                # Avoid division by zero
                bg_b_mean = max(bg_b_mean, 1)
                bg_g_mean = max(bg_g_mean, 1) 
                bg_r_mean = max(bg_r_mean, 1)
                
                contrast_b_values = (flake_b_values / bg_b_mean) - 1
                contrast_g_values = (flake_g_values / bg_g_mean) - 1
                contrast_r_values = (flake_r_values / bg_r_mean) - 1
            else:
                # Fallback if no background pixels found
                contrast_b_values = flake_b_values.astype(np.float32)
                contrast_g_values = flake_g_values.astype(np.float32)
                contrast_r_values = flake_r_values.astype(np.float32)
            
            # Save individual contrast data for this image
            if self.image_files and self.current_index >= 0:
                current_image = self.image_files[self.current_index]
                image_name = os.path.splitext(os.path.basename(current_image))[0]
                # Save with correct RGB order (convert from BGR to RGB)
                self.save_individual_contrast_data(image_name, contrast_r_values, contrast_g_values, contrast_b_values)
            
            # Add contrast values to cumulative arrays (convert BGR order to RGB order)
            self.cumulative_r_values = np.concatenate([self.cumulative_r_values, contrast_r_values])
            self.cumulative_g_values = np.concatenate([self.cumulative_g_values, contrast_g_values]) 
            self.cumulative_b_values = np.concatenate([self.cumulative_b_values, contrast_b_values])
            
            # Rebuild cumulative data from all individual files to avoid duplication
            self.rebuild_cumulative_from_individual_files(project_folder)
            
            # Save updated cumulative data to project folder
            data_path = os.path.join(project_folder, "rgb_contrast_data.npz")
            np.savez(data_path, 
                    r=self.cumulative_r_values, 
                    g=self.cumulative_g_values, 
                    b=self.cumulative_b_values)
            
            # Update plots based on current mode
            if self.btn_toggle_plots.isChecked():
                # Show single image data
                self.update_single_image_plots()
            else:
                # Show cumulative data
                self.update_rgb_contrast_plots()
            
            return True  # Successfully processed mask data
        else:
            return False  # No flake regions found
    
    def clear_markers(self):
        """Clear all markers from the current image"""
        self.image_viewer.clear_markers()
        
        # Ensure mask toggle button state remains correct based on existing masks
        if self.image_files and 0 <= self.current_index < len(self.image_files):
            current_image = self.image_files[self.current_index]
            has_mask = current_image in self.mask_files
            self.btn_toggle_mask.setEnabled(has_mask)
        
        self.status_label.setText("Markers cleared")
    
    def find_existing_masks(self, project_folder):
        """Find existing mask files for loaded images - robust against extension mismatches"""
        self.mask_files.clear()
        masks_dir = os.path.join(project_folder, "masks")
        
        if not os.path.exists(masks_dir):
            return
            
        # Get all mask files in the directory
        mask_files_in_dir = os.listdir(masks_dir)
        
        for image_path in self.image_files:
            image_name = os.path.basename(image_path)
            image_basename = os.path.splitext(image_name)[0]  # Remove extension
            
            # Try to find matching mask file with flexible extension matching
            mask_path = None
            
            # First try: exact filename match
            if image_name in mask_files_in_dir:
                mask_path = os.path.join(masks_dir, image_name)
            else:
                # Second try: same basename with different extensions
                possible_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
                for ext in possible_extensions:
                    potential_mask = image_basename + ext
                    if potential_mask in mask_files_in_dir:
                        mask_path = os.path.join(masks_dir, potential_mask)
                        break
                
                # Third try: check for _mask suffix
                if mask_path is None:
                    for ext in possible_extensions:
                        potential_mask = image_basename + "_mask" + ext
                        if potential_mask in mask_files_in_dir:
                            mask_path = os.path.join(masks_dir, potential_mask)
                            break
            
            if mask_path and os.path.exists(mask_path):
                self.mask_files[image_path] = mask_path
    
    def find_existing_semantic_masks(self, project_folder):
        """Find existing semantic mask files for loaded images - same name as image"""
        self.semantic_mask_files.clear()
        semantic_masks_dir = os.path.join(project_folder, "semantic_masks")
        
        if not os.path.exists(semantic_masks_dir):
            return
            
        # Get all semantic mask files in the directory
        semantic_files_in_dir = os.listdir(semantic_masks_dir)
        
        for image_path in self.image_files:
            image_name = os.path.basename(image_path)
            image_basename = os.path.splitext(image_name)[0]  # Remove extension
            
            # Try to find matching semantic mask file with same name as image
            semantic_path = None
            
            # First try: exact filename match
            if image_name in semantic_files_in_dir:
                semantic_path = os.path.join(semantic_masks_dir, image_name)
            else:
                # Second try: same basename with different extensions
                possible_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
                for ext in possible_extensions:
                    potential_semantic = image_basename + ext
                    if potential_semantic in semantic_files_in_dir:
                        semantic_path = os.path.join(semantic_masks_dir, potential_semantic)
                        break
            
            if semantic_path and os.path.exists(semantic_path):
                self.semantic_mask_files[image_path] = semantic_path
                
    def toggle_mask_view(self):
        """Toggle between showing the original image and its colored instance mask"""
        if not self.image_files or self.current_index < 0:
            return
            
        current_image = self.image_files[self.current_index]
        if current_image not in self.mask_files:
            self.btn_toggle_mask.setChecked(False)
            return
            
        self.showing_mask = self.btn_toggle_mask.isChecked()
        if self.showing_mask:
            # Display colored instance mask with original image blending
            original_image_path = current_image
            mask_path = self.mask_files[current_image]
            
            # Try to display as colored instance mask
            success = self.image_viewer.display_instance_mask(mask_path, original_image_path)
            
            if not success:
                # Fallback to regular mask display if instance mask loading fails
                self.image_viewer.set_image(self.mask_files[current_image])
        else:
            # Apply vignette correction to original images
            self.image_viewer.set_image(current_image, self.flatfield_path)
    
    def toggle_semantic_mask_view(self):
        """Toggle between showing the original image and its colored semantic mask"""
        if not self.image_files or self.current_index < 0:
            return
            
        current_image = self.image_files[self.current_index]
        if current_image not in self.semantic_mask_files:
            self.btn_toggle_semantic.setChecked(False)
            return
            
        self.showing_semantic_mask = self.btn_toggle_semantic.isChecked()
        if self.showing_semantic_mask:
            # Display colored semantic mask with original image blending
            original_image_path = current_image
            semantic_mask_path = self.semantic_mask_files[current_image]
            
            # Try to display as colored semantic mask
            success = self.image_viewer.display_semantic_mask(semantic_mask_path, original_image_path)
            
            if not success:
                # Fallback to regular mask display if semantic mask loading fails
                self.image_viewer.set_image(self.semantic_mask_files[current_image])
        else:
            # Apply vignette correction to original images
            self.image_viewer.set_image(current_image, self.flatfield_path)
    
    def toggle_plot_mode(self):
        """Toggle between cumulative and single image contrast plots"""
        if self.btn_toggle_plots.isChecked():
            # Show single image plots
            self.btn_toggle_plots.setText("Show Cumulative")
            self.update_single_image_plots()
        else:
            # Show cumulative plots
            self.btn_toggle_plots.setText("Show Single Image")
            if len(self.cumulative_r_values) > 0:
                self.update_rgb_contrast_plots()
            else:
                self.clear_plots()

    def clear_plots(self):
        """Clear all plots"""
        for plot in [self.plot1, self.plot2, self.plot3]:
            plot.clear_plot()

    def update_single_image_plots(self):
        """Update plots with single image contrast data"""
        if not self.image_files or self.current_index < 0:
            self.clear_plots()
            return
            
        # Get current image path and check if it has contrast data
        current_image = self.image_files[self.current_index]
        image_name = os.path.splitext(os.path.basename(current_image))[0]
        
        # Get project folder
        project_folder = os.path.dirname(self.image_folder)
        contrast_dir = os.path.join(project_folder, "contrast_data")
        contrast_file = os.path.join(contrast_dir, f"{image_name}_contrast.npz")
        
        if not os.path.exists(contrast_file):
            # No contrast data for current image, clear plots
            self.clear_plots()
            return
            
        try:
            # Load single image contrast data
            data = np.load(contrast_file)
            r_values = data['r']
            g_values = data['g']
            b_values = data['b']
            
            # Clear existing plots and colorbars
            self.clear_and_setup_plots([self.plot1, self.plot2, self.plot3])

            # Create histograms using helper method
            contrast_min, contrast_max = -1, 0.5  # Typical contrast range
            hist_rg, hist_gb, hist_br = self.create_contrast_histograms(
                r_values, g_values, b_values, 
                bins=200, sigma=3, range_min=contrast_min, range_max=contrast_max
            )

            # Update plots using helper method
            extent = [contrast_min, contrast_max, contrast_min, contrast_max]
            self.update_plot_with_histogram(
                self.plot1, hist_rg, extent, 
                f"Red vs Green ({image_name})", "Red", "Green"
            )
            self.update_plot_with_histogram(
                self.plot2, hist_gb, extent,
                f"Green vs Blue ({image_name})", "Green", "Blue"
            )
            self.update_plot_with_histogram(
                self.plot3, hist_br, extent,
                f"Blue vs Red ({image_name})", "Blue", "Red"
            )
            
        except Exception as e:
            print(f"Error loading single image contrast data for {image_name}: {e}")
            self.clear_plots()

    def update_rgb_contrast_plots(self):
        """Update plots with cumulative RGB contrast data"""
        if len(self.cumulative_r_values) == 0:
            self.clear_plots()
            return
            
        # Clear existing plots and colorbars
        self.clear_and_setup_plots([self.plot1, self.plot2, self.plot3])

        # Create histograms using helper method
        contrast_min, contrast_max = -1, 0.5  # Typical contrast range
        hist_rg, hist_gb, hist_br = self.create_contrast_histograms(
            self.cumulative_r_values, self.cumulative_g_values, self.cumulative_b_values,
            bins=200, sigma=3, range_min=contrast_min, range_max=contrast_max
        )

        # Update plots using helper method
        extent = [contrast_min, contrast_max, contrast_min, contrast_max]
        self.update_plot_with_histogram(
            self.plot1, hist_rg, extent,
            "Red vs Green Contrast (Cumulative)", "Red", "Green"
        )
        self.update_plot_with_histogram(
            self.plot2, hist_gb, extent,
            "Green vs Blue Contrast (Cumulative)", "Green", "Blue"
        )
        self.update_plot_with_histogram(
            self.plot3, hist_br, extent,
            "Blue vs Red Contrast (Cumulative)", "Blue", "Red"
        )

    def create_contrast_histograms(self, r_values, g_values, b_values, bins=200, sigma=1, range_min=-1, range_max=3):
        """Create 2D histograms for contrast data with consistent processing"""
        # Red vs Green
        hist_rg, _, _ = np.histogram2d(
            r_values, g_values, bins=bins,
            range=[[range_min, range_max], [range_min, range_max]]
        )
        hist_rg = gaussian_filter(hist_rg, sigma=sigma)
        hist_rg = np.log(hist_rg + 1)
        
        # Green vs Blue
        hist_gb, _, _ = np.histogram2d(
            g_values, b_values, bins=bins,
            range=[[range_min, range_max], [range_min, range_max]]
        )
        hist_gb = gaussian_filter(hist_gb, sigma=sigma)
        hist_gb = np.log(hist_gb + 1)
        
        # Blue vs Red
        hist_br, _, _ = np.histogram2d(
            b_values, r_values, bins=bins,
            range=[[range_min, range_max], [range_min, range_max]]
        )
        hist_br = gaussian_filter(hist_br, sigma=sigma)
        hist_br = np.log(hist_br + 1)
        
        return hist_rg, hist_gb, hist_br

    def clear_and_setup_plots(self, plots):
        """Clear and setup plots for new data"""
        for plot in plots:
            plot.fig.clear()
            plot.axes = plot.fig.add_subplot(111)

    def update_plot_with_histogram(self, plot, hist, extent, title, xlabel, ylabel):
        """Update a single plot with histogram data"""
        plot.axes.clear()
        im = plot.axes.imshow(hist.T, origin='lower', aspect='auto',
                            extent=extent, cmap=cm.plasma)
        
        # Set titles and labels with smaller font sizes
        plot.axes.set_title(title, fontsize=9)
        plot.axes.set_xlabel(xlabel, fontsize=8)
        plot.axes.set_ylabel(ylabel, fontsize=8)
        
        # Adjust tick label sizes
        plot.axes.tick_params(axis='both', which='major', labelsize=7)
        
        # Add colorbar with proper spacing
        cbar = plot.fig.colorbar(im, ax=plot.axes, shrink=0.8)
        cbar.ax.tick_params(labelsize=7)
        
        # Ensure proper spacing for labels
        plot.fig.subplots_adjust(left=0.15, bottom=0.15, right=0.85, top=0.85)
        plot.draw()

    def save_individual_contrast_data(self, image_name, r_values, g_values, b_values):
        """Save contrast data for a specific image"""
        if not self.image_files:
            return
            
        # Get project folder
        project_folder = os.path.dirname(self.image_folder)
        contrast_dir = os.path.join(project_folder, "contrast_data")
        
        # Create directory if it doesn't exist
        os.makedirs(contrast_dir, exist_ok=True)
        
        # Save individual contrast data
        contrast_file = os.path.join(contrast_dir, f"{image_name}_contrast.npz")
        np.savez(contrast_file, r=r_values, g=g_values, b=b_values)

    def load_cumulative_contrast_data(self, project_folder):
        """Load cumulative contrast data from files"""
        self.cumulative_r_values = np.array([], dtype=np.float32)
        self.cumulative_g_values = np.array([], dtype=np.float32)
        self.cumulative_b_values = np.array([], dtype=np.float32)
        
        data_path = os.path.join(project_folder, "rgb_contrast_data.npz")
        
        if os.path.exists(data_path):
            try:
                # Load cumulative data
                data = np.load(data_path)
                self.cumulative_r_values = data['r']
                self.cumulative_g_values = data['g']
                self.cumulative_b_values = data['b']
                
                # Update status
                self.status_label.setText("Cumulative data loaded")
                return
            except Exception as e:
                print(f"Error loading cumulative contrast data: {e}")
        
        # If cumulative file doesn't exist, try to rebuild from individual files
        self.rebuild_cumulative_from_individual_files(project_folder)
        
        if len(self.cumulative_r_values) > 0:
            # Save the rebuilt cumulative data
            np.savez(data_path, 
                    r=self.cumulative_r_values, 
                    g=self.cumulative_g_values, 
                    b=self.cumulative_b_values)
            
            individual_files_count = len([f for f in os.listdir(os.path.join(project_folder, "contrast_data")) 
                                        if f.endswith("_contrast.npz")]) if os.path.exists(os.path.join(project_folder, "contrast_data")) else 0
            self.status_label.setText(f"Cumulative data rebuilt from {individual_files_count} individual files")
            return
        
        # No data found at all
        self.status_label.setText("No contrast data found")
        self.clear_plots()
    
    def rebuild_cumulative_from_individual_files(self, project_folder):
        """Rebuild cumulative contrast data from all individual files"""
        self.cumulative_r_values = np.array([], dtype=np.float32)
        self.cumulative_g_values = np.array([], dtype=np.float32)
        self.cumulative_b_values = np.array([], dtype=np.float32)
        
        contrast_dir = os.path.join(project_folder, "contrast_data")
        if not os.path.exists(contrast_dir):
            return
            
        all_r_values = []
        all_g_values = []
        all_b_values = []
        
        # Look for individual contrast files
        for file in os.listdir(contrast_dir):
            if file.endswith("_contrast.npz"):
                contrast_file = os.path.join(contrast_dir, file)
                try:
                    data = np.load(contrast_file)
                    all_r_values.append(data['r'])
                    all_g_values.append(data['g'])
                    all_b_values.append(data['b'])
                except Exception as e:
                    print(f"Error loading individual contrast file {file}: {e}")
        
        if all_r_values:  # If we found any individual files
            # Combine all individual data into cumulative arrays
            self.cumulative_r_values = np.concatenate(all_r_values)
            self.cumulative_g_values = np.concatenate(all_g_values)
            self.cumulative_b_values = np.concatenate(all_b_values)

    def open_class_annotator(self):
        """Open the class annotator dialog"""
        if not self.image_files:
            QMessageBox.warning(self, "Warning", "No project folder selected")
            return
            
        # Get project folder from image folder
        project_folder = os.path.dirname(self.image_folder)
        
        # Check if instance masks exist
        masks_dir = os.path.join(project_folder, "masks")
        if not os.path.exists(masks_dir) or not os.listdir(masks_dir):
            QMessageBox.warning(self, "Warning", 
                              "No instance masks found. Please create some instance masks first.")
            return
            
        try:
            from gui.class_annotator_dialog import ClassAnnotatorDialog
            dialog = ClassAnnotatorDialog(self, project_folder)
            # Force maximized state before showing
            dialog.setWindowState(Qt.WindowMaximized)
            dialog.showMaximized()
            dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open class annotator: {e}")
            
    def open_amm_training(self):
        """Open the AMM training dashboard"""
        if not self.image_folder:
            QMessageBox.warning(self, "Warning", "Please select a project folder first.")
            return
            
        # Get the actual project folder (parent of image folder)
        project_folder = os.path.dirname(self.image_folder)
            
        # Check if semantic masks exist
        semantic_masks_dir = os.path.join(project_folder, "semantic_masks")
        if not os.path.exists(semantic_masks_dir):
            QMessageBox.warning(self, "Warning", 
                              "No semantic masks found. Please use the Class Annotator to create semantic masks first.")
            return
            
        # Check if there are any semantic mask files
        semantic_files = [f for f in os.listdir(semantic_masks_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not semantic_files:
            QMessageBox.warning(self, "Warning", 
                              "No semantic mask files found. Please annotate classes first using the Class Annotator.")
            return
            
        try:
            from gui.amm_training_dialog import AMMTrainingDialog
            dialog = AMMTrainingDialog(self, project_folder)
            dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open AMM training dashboard: {e}")
    
    def open_gmm_training(self):
        """Open the GMM training dialog"""
        # Get project folder from current image folder
        if not self.image_folder:
            QMessageBox.warning(self, "No Folder Selected", 
                              "Please select a project folder first.")
            return
        
        # Get the material folder (parent of image folder)
        # e.g., if image_folder is Materials/Graphene/images, project_folder is Materials/Graphene
        project_folder = os.path.dirname(self.image_folder)
        
        # Check if instance masks exist
        masks_dir = os.path.join(project_folder, "masks")
        if not os.path.exists(masks_dir):
            QMessageBox.warning(self, "Warning", 
                              "No instance masks found. Please create instance masks first.")
            return
            
        # Check if there are any mask files
        mask_files = [f for f in os.listdir(masks_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not mask_files:
            QMessageBox.warning(self, "Warning", 
                              "No mask files found. Please create instance masks first.")
            return
            
        try:
            from gui.gmm_training_dialog import GMMTrainingDialog
            dialog = GMMTrainingDialog(self, project_folder)
            dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open Training dashboard: {e}")
    
    def open_m2f_training(self):
        """Open the M2F training dialog"""
        # Get project folder from current image folder
        if not self.image_folder:
            QMessageBox.warning(self, "No Folder Selected", 
                              "Please select a project folder first.")
            return
        
        # Find project root by going up from image folder
        project_folder = self.image_folder
        while project_folder and not os.path.basename(project_folder) == "Materials":
            project_folder = os.path.dirname(project_folder)
        
        if project_folder:
            project_folder = os.path.dirname(project_folder)  # Go up one more level from Materials
        else:
            project_folder = os.path.dirname(self.image_folder)
        
        try:
            from gui.m2f_training_dialog import M2FTrainingDialog
            dialog = M2FTrainingDialog(project_folder, self)
            dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open M2F training dialog: {e}")
    
    def show_help(self):
        """Show help dialog for the main window"""
        show_help_dialog(self, "main_window", "Main Window Help")

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()