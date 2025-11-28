import os
import cv2
from PyQt5.QtWidgets import (QLabel, QSizePolicy)
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QPixmap, QImage, QPainter
import numpy as np

# Import vignette removal function and GMM functions
from zpreprocessor_functions import remove_vignette

class ImageViewer(QLabel):
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("QLabel { background-color: #f0f0f0; border: 1px solid #ccc; }")
        self.setScaledContents(False)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.original_pixmap = None
        self.original_image_path = None  # Store path to always access original image
        self.markers = []  # List of (x, y, color) tuples
        self.current_pixmap = None
        
        # Performance optimization: Cache for expensive vignette corrections
        self._vignette_cache = {}  # Cache for vignette corrected images
        
        # Enable mouse tracking
        self.setMouseTracking(True)
        
    def set_image(self, image_path, flatfield_path=None):
        """Load and display an image with optional vignette correction and caching"""
        if os.path.exists(image_path):
            # Store the original image path for contrast calculations
            self.original_image_path = image_path
            
            if flatfield_path and os.path.exists(flatfield_path):
                # Check cache first for vignette corrected image
                cache_key = (image_path, flatfield_path)
                if cache_key in self._vignette_cache:
                    self.original_pixmap = self._vignette_cache[cache_key]
                else:
                    # Load image as BGR for vignette correction
                    image_bgr = cv2.imread(image_path)
                    flatfield_bgr = cv2.imread(flatfield_path)
                    
                    if image_bgr is not None and flatfield_bgr is not None:
                        # Apply vignette correction (expensive operation)
                        corrected_image = remove_vignette(image_bgr, flatfield_bgr)
                        
                        # Convert BGR back to RGB for Qt
                        corrected_image_rgb = cv2.cvtColor(corrected_image, cv2.COLOR_BGR2RGB)
                        
                        # Convert to QPixmap
                        height, width, channel = corrected_image_rgb.shape
                        bytes_per_line = 3 * width
                        q_image = QImage(corrected_image_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
                        self.original_pixmap = QPixmap.fromImage(q_image)
                        
                        # Cache the result (limit cache size to prevent memory issues)
                        if len(self._vignette_cache) > 10:  # Keep only 10 most recent
                            # Remove oldest entry
                            oldest_key = next(iter(self._vignette_cache))
                            del self._vignette_cache[oldest_key]
                        self._vignette_cache[cache_key] = self.original_pixmap
                    else:
                        # Fallback to normal loading if vignette correction fails
                        self.original_pixmap = QPixmap(image_path)
            else:
                # Normal loading without vignette correction
                self.original_pixmap = QPixmap(image_path)
            
            self.update_display()
            return True
        return False
    
    def update_display(self):
        """Update the displayed image with proper scaling and markers"""
        if self.original_pixmap:
            # Scale image to fit the widget while maintaining aspect ratio
            scaled_pixmap = self.original_pixmap.scaled(
                self.size(), 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
            
            # Create a copy of the scaled pixmap for drawing markers
            self.current_pixmap = scaled_pixmap.copy()
            
            if self.markers:
                # Draw markers on the scaled image
                painter = QPainter(self.current_pixmap)
                painter.setRenderHint(QPainter.Antialiasing)
                
                # Get the display rect for scaling marker positions
                rect = self.get_display_rect()
                scale_x = rect.width() / self.original_pixmap.width()
                scale_y = rect.height() / self.original_pixmap.height()
                
                for x, y, color in self.markers:
                    # Scale marker position to display coordinates
                    display_x = int(x * scale_x)
                    display_y = int(y * scale_y)
                    
                    # Set marker color
                    if color == "red":
                        painter.setPen(Qt.red)
                        painter.setBrush(Qt.red)
                    else:
                        painter.setPen(Qt.green)
                        painter.setBrush(Qt.green)
                    
                    # Draw marker
                    painter.drawEllipse(display_x - 2, display_y - 2, 5, 5)
                
                painter.end()
            
            self.setPixmap(self.current_pixmap)
    
    def resizeEvent(self, event):
        """Handle resize events to update image scaling"""
        super().resizeEvent(event)
        self.update_display()
    
    def mousePressEvent(self, event):
        """Handle mouse press events for adding markers"""
        if not self.original_pixmap:
            return
            
        # Get click position relative to the actual image
        pos = self.mapImagePos(event.pos())
        if pos is None:
            return
            
        # Add marker based on which button was pressed
        if event.button() == Qt.LeftButton:
            self.markers.append((pos.x(), pos.y(), "green"))
        elif event.button() == Qt.RightButton:
            self.markers.append((pos.x(), pos.y(), "red"))
            
        # Create watershed mask and update display
        if len(self.markers) >= 2:  # Need at least one marker of each type
            has_red = any(color == "red" for _, _, color in self.markers)
            has_green = any(color == "green" for _, _, color in self.markers)
            if has_red and has_green:
                self.create_watershed_mask()
            else:
                self.update_display()
        else:
            self.update_display()
    
    def mapImagePos(self, pos):
        """Map window coordinates to image coordinates"""
        if not self.current_pixmap:
            return None
            
        # Get the image display rect
        rect = self.get_display_rect()
        if not rect.contains(pos):
            return None
            
        # Calculate relative position within the image
        x = (pos.x() - rect.x()) * (self.original_pixmap.width() / rect.width())
        y = (pos.y() - rect.y()) * (self.original_pixmap.height() / rect.height())
        
        return QPoint(int(x), int(y))
    
    def get_display_rect(self):
        """Get the rectangle where the image is actually displayed"""
        if not self.current_pixmap:
            return None
            
        # Calculate the display rectangle maintaining aspect ratio
        w = self.width()
        h = self.height()
        
        image_ratio = self.original_pixmap.width() / self.original_pixmap.height()
        widget_ratio = w / h
        
        if widget_ratio > image_ratio:
            new_w = int(h * image_ratio)
            new_h = h
            x = (w - new_w) // 2
            y = 0
        else:
            new_w = w
            new_h = int(w / image_ratio)
            x = 0
            y = (h - new_h) // 2
            
        return self.rect().adjusted(x, y, -(w - new_w - x), -(h - new_h - y))
    
    def clear_markers(self):
        """Clear all markers and redraw"""
        self.markers.clear()
        if self.original_pixmap:
            self.current_pixmap = self.original_pixmap.scaled(
                self.size(), 
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.setPixmap(self.current_pixmap)
    
    def create_watershed_mask(self):
        """Create an instance mask using watershed algorithm based on markers"""
        if not self.original_pixmap or not self.markers:
            return None
            
        # Convert QPixmap to numpy array
        image = self.pixmap_to_numpy(self.original_pixmap)
        
        # Create markers image
        markers = np.zeros(image.shape[:2], dtype=np.int32)
        
        # Convert marker positions to original image coordinates
        for x, y, color in self.markers:
            # Use 1 for background (red) and 2 for flakes (green)
            marker_value = 2 if color == "green" else 1
            markers[int(y), int(x)] = marker_value
        
        # Run watershed algorithm
        cv2.watershed(image, markers)
        
        # Create binary mask (255 for flakes, 0 for background)
        binary_mask = np.zeros(markers.shape, dtype=np.uint8)
        binary_mask[markers == 2] = 255  # Flakes
        
        # Convert binary mask to instance mask
        from scripts.mask_functions import binary_to_instance_mask
        instance_mask = binary_to_instance_mask(binary_mask)
        
        # Update display to show segmentation with colored instances
        overlay = self.create_colored_instance_visualization(instance_mask, image, blend_with_original=False)
        
        # Convert back to QPixmap and display
        self.display_watershed_result(overlay)
        
        return instance_mask
    
    def pixmap_to_numpy(self, pixmap):
        """Convert QPixmap to numpy array"""
        # Convert QPixmap to QImage
        image = pixmap.toImage()
        # Convert QImage to numpy array
        width = image.width()
        height = image.height()
        ptr = image.bits()
        ptr.setsize(height * width * 4)  # 4 bytes per pixel (RGBA)
        arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))
        # Convert RGBA to BGR (OpenCV format)
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    
    def display_watershed_result(self, image):
        """Display the watershed segmentation result"""
        height, width = image.shape[:2]
        bytes_per_line = 3 * width
        q_img = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.current_pixmap = QPixmap.fromImage(q_img)
        
        # Scale the result to match the current display size
        scaled_pixmap = self.current_pixmap.scaled(
            self.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.current_pixmap = scaled_pixmap
        self.setPixmap(self.current_pixmap)
    
    def display_instance_mask(self, mask_path, original_image_path=None):
        """Display an instance mask with colored instances"""
        if not os.path.exists(mask_path):
            return False
            
        # Load the instance mask
        instance_mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if instance_mask is None:
            return False
            
        # Load original image for blending (optional)
        if original_image_path and os.path.exists(original_image_path):
            original_image = cv2.imread(original_image_path)
            if original_image is not None:
                original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            else:
                original_image = None
        else:
            original_image = None
            
        # Create colored visualization
        colored_mask = self.create_colored_instance_visualization(instance_mask, original_image, blend_with_original=True)
        
        # Convert to QPixmap and display
        height, width = colored_mask.shape[:2]
        bytes_per_line = 3 * width
        q_img = QImage(colored_mask.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        # Store the blended visualization as current pixmap, but don't overwrite original_pixmap
        self.current_pixmap = QPixmap.fromImage(q_img)
        
        # Scale and display the blended result
        scaled_pixmap = self.current_pixmap.scaled(
            self.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.setPixmap(scaled_pixmap)
        return True
    
    def display_semantic_mask(self, semantic_mask_path, original_image_path=None):
        """Display a semantic mask with colored classes"""
        if not os.path.exists(semantic_mask_path):
            return False
            
        # Load the semantic mask
        semantic_mask = cv2.imread(semantic_mask_path, cv2.IMREAD_UNCHANGED)
        if semantic_mask is None:
            return False
            
        # Load original image for blending (optional)
        if original_image_path and os.path.exists(original_image_path):
            original_image = cv2.imread(original_image_path)
            if original_image is not None:
                original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            else:
                original_image = None
        else:
            original_image = None
            
        # Create colored visualization
        colored_mask = self.create_colored_semantic_visualization(semantic_mask, original_image, blend_with_original=True)
        
        # Convert to QPixmap and display
        height, width = colored_mask.shape[:2]
        bytes_per_line = 3 * width
        q_img = QImage(colored_mask.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        # Store the blended visualization as current pixmap
        self.current_pixmap = QPixmap.fromImage(q_img)
        
        # Scale and display the blended result
        scaled_pixmap = self.current_pixmap.scaled(
            self.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.setPixmap(scaled_pixmap)
        return True
    
    def create_colored_semantic_visualization(self, semantic_mask, original_image=None, blend_with_original=True):
        """Create a colored visualization of a semantic mask
        
        Args:
            semantic_mask: The semantic mask with class IDs
            original_image: The original image for blending (optional)
            blend_with_original: If True, blend with original image
        """
        # Color palette for classes (using tab10 colormap style)
        class_colors = [
            [0, 0, 0],         # Class 0 - Background (black)
            [31, 119, 180],    # Class 1 - Blue
            [255, 127, 14],    # Class 2 - Orange
            [44, 160, 44],     # Class 3 - Green
            [214, 39, 40],     # Class 4 - Red
            [148, 103, 189],   # Class 5 - Purple
            [140, 86, 75],     # Class 6 - Brown
            [227, 119, 194],   # Class 7 - Pink
            [127, 127, 127],   # Class 8 - Gray
            [188, 189, 34],    # Class 9 - Olive
            [23, 190, 207],    # Class 10 - Cyan
        ]
        
        # Create output image
        if original_image is not None and blend_with_original:
            colored_mask = original_image.copy()
        else:
            colored_mask = np.zeros((*semantic_mask.shape[:2], 3), dtype=np.uint8)
        
        # Get unique class IDs
        unique_classes = np.unique(semantic_mask)
        
        if blend_with_original and original_image is not None:
            # Blend mode: Semi-transparent overlay
            for class_id in unique_classes:
                if class_id == 0:  # Skip background
                    continue
                    
                color_idx = class_id % len(class_colors)
                class_pixels = semantic_mask == class_id
                class_color = np.array(class_colors[color_idx], dtype=np.uint8)
                
                # Apply semi-transparent overlay (70% original + 30% color)
                colored_mask[class_pixels] = cv2.addWeighted(
                    colored_mask[class_pixels], 0.7,
                    np.full_like(colored_mask[class_pixels], class_color), 0.3, 0
                )
            
            # Add white boundaries between classes
            kernel = np.ones((3, 3), np.uint8)
            boundaries = cv2.morphologyEx(semantic_mask.astype(np.uint8), cv2.MORPH_GRADIENT, kernel)
            boundaries = boundaries > 0
            colored_mask[boundaries] = [255, 255, 255]
        else:
            # Pure colored classes without blending
            for class_id in unique_classes:
                color_idx = class_id % len(class_colors)
                class_pixels = semantic_mask == class_id
                colored_mask[class_pixels] = class_colors[color_idx]
        
        return colored_mask
    
    def create_colored_instance_visualization(self, instance_mask, original_image=None, blend_with_original=True):
        """Create a colored visualization of an instance mask
        
        Args:
            instance_mask: The instance mask with unique IDs
            original_image: The original image for blending (optional)
            blend_with_original: If True, blend with original image using addWeighted.
                               If False, show pure colored instances for clearer visualization.
        """
        # Color palette for instances (same as watershed)
        class_colors = [
            [0, 0, 0],        # Background (black)
            [255, 127, 14],   # Orange
            [44, 160, 44],    # Green  
            [214, 39, 40],    # Red
            [148, 103, 189],  # Purple
            [140, 86, 75],    # Brown
            [227, 119, 194],  # Pink
            [127, 127, 127],  # Gray
            [188, 189, 34],   # Olive
            [23, 190, 207],   # Cyan
        ]
        
        # Create colored visualization
        if original_image is not None:
            colored_mask = original_image.copy()
        else:
            colored_mask = np.zeros((instance_mask.shape[0], instance_mask.shape[1], 3), dtype=np.uint8)
        
        # Get unique instance IDs
        unique_instances = np.unique(instance_mask)
        unique_instances = unique_instances[unique_instances > 0]  # Remove background
        
        if blend_with_original:
            # Show Mask mode: Full blended overlay
            overlay = np.zeros_like(colored_mask)
            
            # Apply colors to each instance in overlay
            for instance_id in unique_instances:
                color_idx = ((instance_id - 1) % (len(class_colors) - 1)) + 1
                instance_pixels = instance_mask == instance_id
                overlay[instance_pixels] = class_colors[color_idx]
            
            # Blend the entire overlay with original image
            colored_mask = cv2.addWeighted(colored_mask, 0.7, overlay, 0.3, 0)
            
            # Add white boundaries between instances
            kernel = np.ones((3, 3), np.uint8)
            boundaries = cv2.morphologyEx(instance_mask.astype(np.uint8), cv2.MORPH_GRADIENT, kernel)
            boundaries = boundaries > 0
            colored_mask[boundaries] = [255, 255, 255]
            
        else:
            # Marker placement mode: Light tint + white boundaries
            for instance_id in unique_instances:
                color_idx = ((instance_id - 1) % (len(class_colors) - 1)) + 1
                instance_pixels = instance_mask == instance_id
                instance_color = np.array(class_colors[color_idx], dtype=np.uint8)
                
                # Apply light tint (90% original + 10% color for subtle effect)
                colored_mask[instance_pixels] = cv2.addWeighted(
                    colored_mask[instance_pixels], 0.9,
                    np.full_like(colored_mask[instance_pixels], instance_color), 0.1, 0
                )
            
            # Add white boundaries between instances
            kernel = np.ones((3, 3), np.uint8)
            boundaries = cv2.morphologyEx(instance_mask.astype(np.uint8), cv2.MORPH_GRADIENT, kernel)
            boundaries = boundaries > 0
            colored_mask[boundaries] = [255, 255, 255]
        
        return colored_mask
    
    def get_original_image_for_analysis(self):
        """Get the original image data for contrast analysis, bypassing any display modifications"""
        if self.original_image_path and os.path.exists(self.original_image_path):
            # Always load fresh from the original file to ensure no modifications
            original_image = cv2.imread(self.original_image_path)
            return original_image
        else:
            # Fallback to pixmap conversion if path not available
            if self.original_pixmap:
                return self.pixmap_to_numpy(self.original_pixmap)
        return None