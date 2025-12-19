import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QSpinBox, QDoubleSpinBox, QMessageBox, QProgressBar,
                             QProgressDialog, QApplication)
from PyQt5.QtCore import Qt, QEvent, QTimer, QThread, pyqtSignal
from gui.help_dialog import show_help_dialog
from scipy.ndimage import gaussian_filter
import cv2


class ContrastLoaderThread(QThread):
    """Thread for loading contrast data without blocking the UI"""
    progress_update = pyqtSignal(str)
    finished = pyqtSignal(object, object)  # (instance_contrasts, instance_classifiers)
    error = pyqtSignal(str)
    
    def __init__(self, image_directory, mask_directory, flatfield_path, use_flatfield):
        super().__init__()
        self.image_directory = image_directory
        self.mask_directory = mask_directory
        self.flatfield_path = flatfield_path
        self.use_flatfield = use_flatfield
        
    def run(self):
        try:
            self.progress_update.emit("Loading contrast data from images...")
            
            # Add scripts to path
            scripts_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts")
            if scripts_path not in sys.path:
                sys.path.append(scripts_path)
            
            from scripts.preprocessor_functions import get_instance_contrasts_from_dir
            
            instance_contrasts, instance_classifiers = get_instance_contrasts_from_dir(
                self.image_directory, 
                self.mask_directory,
                self.flatfield_path if self.use_flatfield else None,
                self.use_flatfield,
                min_instance_size=10
            )
            
            self.finished.emit(instance_contrasts, instance_classifiers)
            
        except Exception as e:
            self.error.emit(str(e))

class ClassAnnotatorDialog(QDialog):
    def __init__(self, parent=None, project_folder=None):
        super().__init__(parent)
        self.project_folder = project_folder
        self.contrast_data = None
        self.current_class = 1
        self.class_assignments = None
        self.selected_points = []
        self.loader_thread = None
        self.progress_dialog = None
        
        # Store axis limits for each color
        self.r_limits = [-1.0, 0.5]
        self.g_limits = [-1.0, 0.5]
        self.b_limits = [-1.0, 0.5]
        
        # Store lasso selectors
        self.lasso_rg = None
        self.lasso_gb = None
        self.lasso_br = None
        
        self.init_ui()
        self.load_contrast_data()
        
        # Install event filter to catch keyboard events
        self.installEventFilter(self)
        
    def eventFilter(self, obj, event):
        """Event filter to catch keyboard events globally for this dialog"""
        if event.type() == QEvent.KeyPress:
            # Handle the key press event directly
            self.handle_key_press(event.key())
            return True  # Event handled
        return super().eventFilter(obj, event)
        
    def init_ui(self):
        self.setWindowTitle("Class Annotator")
        # Set window state to maximized
        self.setWindowState(Qt.WindowMaximized)
        
        # Enable keyboard focus
        self.setFocusPolicy(Qt.StrongFocus)
        
        # Main layout
        main_layout = QVBoxLayout()
        
        # Header with title and help button
        header_layout = QHBoxLayout()
        
        # Title
        title_label = QLabel("🏷️ Class Annotator - Detailed Annotations")
        title_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #2c3e50; margin: 5px;")
        header_layout.addWidget(title_label)
        
        header_layout.addStretch()
        
        # Help button
        help_btn = QPushButton("❓")
        help_btn.setFixedSize(35, 25)
        help_btn.setToolTip("Show Help (F1)")
        help_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 12px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        help_btn.clicked.connect(self.show_help)
        header_layout.addWidget(help_btn)
        
        main_layout.addLayout(header_layout)
        
        # Controls layout
        controls_layout = QHBoxLayout()
        
        # Current class controls
        controls_layout.addWidget(QLabel("Current Class:"))
        self.class_spinbox = QSpinBox()
        self.class_spinbox.setMinimum(0)
        self.class_spinbox.setMaximum(10)
        self.class_spinbox.setValue(1)
        self.class_spinbox.valueChanged.connect(self.on_class_changed)
        controls_layout.addWidget(self.class_spinbox)
        
        # Progress info
        self.progress_label = QLabel("No data loaded")
        controls_layout.addWidget(self.progress_label)
        
        controls_layout.addStretch()
        
        # Action buttons
        self.btn_save = QPushButton("Save Semantic Masks")
        self.btn_save.clicked.connect(self.save_semantic_masks)
        self.btn_save.setEnabled(False)
        controls_layout.addWidget(self.btn_save)
        
        self.btn_close = QPushButton("Close")
        self.btn_close.clicked.connect(self.close)
        controls_layout.addWidget(self.btn_close)
        
        main_layout.addLayout(controls_layout)
        
        # Threshold controls
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("High StdDev Threshold:"))
        self.stddev_threshold_input = QDoubleSpinBox()
        self.stddev_threshold_input.setRange(0.0, 10.0)
        self.stddev_threshold_input.setSingleStep(0.1)
        self.stddev_threshold_input.setValue(0.2)
        self.stddev_threshold_input.setDecimals(2)
        threshold_layout.addWidget(self.stddev_threshold_input)
        
        self.btn_reload = QPushButton("Reload with New Threshold")
        self.btn_reload.clicked.connect(self.load_contrast_data)
        threshold_layout.addWidget(self.btn_reload)
        threshold_layout.addStretch()
        
        main_layout.addLayout(threshold_layout)
        
        # Axis limits controls
        limits_layout = QVBoxLayout()
        limits_label = QLabel("Axis Limits:")
        limits_label.setStyleSheet("font-weight: bold;")
        limits_layout.addWidget(limits_label)
        
        # Red limits
        r_layout = QHBoxLayout()
        r_layout.addWidget(QLabel("Red:"))
        r_layout.addWidget(QLabel("Min:"))
        self.r_min_spinbox = QDoubleSpinBox()
        self.r_min_spinbox.setRange(-5.0, 5.0)
        self.r_min_spinbox.setSingleStep(0.1)
        self.r_min_spinbox.setValue(-1.0)
        self.r_min_spinbox.setDecimals(2)
        self.r_min_spinbox.valueChanged.connect(self.on_r_limits_changed)
        r_layout.addWidget(self.r_min_spinbox)
        r_layout.addWidget(QLabel("Max:"))
        self.r_max_spinbox = QDoubleSpinBox()
        self.r_max_spinbox.setRange(-5.0, 5.0)
        self.r_max_spinbox.setSingleStep(0.1)
        self.r_max_spinbox.setValue(0.5)
        self.r_max_spinbox.setDecimals(2)
        self.r_max_spinbox.valueChanged.connect(self.on_r_limits_changed)
        r_layout.addWidget(self.r_max_spinbox)
        limits_layout.addLayout(r_layout)
        
        # Green limits
        g_layout = QHBoxLayout()
        g_layout.addWidget(QLabel("Green:"))
        g_layout.addWidget(QLabel("Min:"))
        self.g_min_spinbox = QDoubleSpinBox()
        self.g_min_spinbox.setRange(-5.0, 5.0)
        self.g_min_spinbox.setSingleStep(0.1)
        self.g_min_spinbox.setValue(-1.0)
        self.g_min_spinbox.setDecimals(2)
        self.g_min_spinbox.valueChanged.connect(self.on_g_limits_changed)
        g_layout.addWidget(self.g_min_spinbox)
        g_layout.addWidget(QLabel("Max:"))
        self.g_max_spinbox = QDoubleSpinBox()
        self.g_max_spinbox.setRange(-5.0, 5.0)
        self.g_max_spinbox.setSingleStep(0.1)
        self.g_max_spinbox.setValue(0.5)
        self.g_max_spinbox.setDecimals(2)
        self.g_max_spinbox.valueChanged.connect(self.on_g_limits_changed)
        g_layout.addWidget(self.g_max_spinbox)
        limits_layout.addLayout(g_layout)
        
        # Blue limits
        b_layout = QHBoxLayout()
        b_layout.addWidget(QLabel("Blue:"))
        b_layout.addWidget(QLabel("Min:"))
        self.b_min_spinbox = QDoubleSpinBox()
        self.b_min_spinbox.setRange(-5.0, 5.0)
        self.b_min_spinbox.setSingleStep(0.1)
        self.b_min_spinbox.setValue(-1.0)
        self.b_min_spinbox.setDecimals(2)
        self.b_min_spinbox.valueChanged.connect(self.on_b_limits_changed)
        b_layout.addWidget(self.b_min_spinbox)
        b_layout.addWidget(QLabel("Max:"))
        self.b_max_spinbox = QDoubleSpinBox()
        self.b_max_spinbox.setRange(-5.0, 5.0)
        self.b_max_spinbox.setSingleStep(0.1)
        self.b_max_spinbox.setValue(0.5)
        self.b_max_spinbox.setDecimals(2)
        self.b_max_spinbox.valueChanged.connect(self.on_b_limits_changed)
        b_layout.addWidget(self.b_max_spinbox)
        limits_layout.addLayout(b_layout)
        
        main_layout.addLayout(limits_layout)
        
        # Plots layout - three plots side by side
        plots_layout = QHBoxLayout()
        
        # Create three matplotlib canvases with larger size
        self.fig_rg = Figure(figsize=(8, 6), dpi=100)
        self.canvas_rg = FigureCanvas(self.fig_rg)
        self.canvas_rg.setFocusPolicy(Qt.NoFocus)  # Prevent stealing focus
        self.canvas_rg.mpl_connect('button_press_event', lambda event: self.setFocus())
        self.ax_rg = self.fig_rg.add_subplot(111)
        plots_layout.addWidget(self.canvas_rg)
        
        self.fig_gb = Figure(figsize=(8, 6), dpi=100)
        self.canvas_gb = FigureCanvas(self.fig_gb)
        self.canvas_gb.setFocusPolicy(Qt.NoFocus)  # Prevent stealing focus
        self.canvas_gb.mpl_connect('button_press_event', lambda event: self.setFocus())
        self.ax_gb = self.fig_gb.add_subplot(111)
        plots_layout.addWidget(self.canvas_gb)
        
        self.fig_br = Figure(figsize=(8, 6), dpi=100)
        self.canvas_br = FigureCanvas(self.fig_br)
        self.canvas_br.setFocusPolicy(Qt.NoFocus)  # Prevent stealing focus
        self.canvas_br.mpl_connect('button_press_event', lambda event: self.setFocus())
        self.ax_br = self.fig_br.add_subplot(111)
        plots_layout.addWidget(self.canvas_br)
        
        main_layout.addLayout(plots_layout)
        
        # Status bar
        self.status_label = QLabel("Ready")
        main_layout.addWidget(self.status_label)
        
        # Add keyboard shortcuts info
        self.shortcuts_label = QLabel("Shortcuts: D/A (±class) | X (background) | S (save) | C (clear all) | 1-9 (quick class) | Space (toggle view)")
        self.shortcuts_label.setStyleSheet("color: gray; font-size: 10px;")
        main_layout.addWidget(self.shortcuts_label)
        
        self.setLayout(main_layout)
        
    def load_contrast_data(self):
        """Load contrast data from all instance masks in the project"""
        if not self.project_folder:
            self.status_label.setText("No project folder specified")
            return
            
        # Get paths
        image_directory = os.path.join(self.project_folder, "images")
        mask_directory = os.path.join(self.project_folder, "masks")
        flatfield_path = os.path.join(self.project_folder, "flatfield.png")
        
        # Check if required directories exist
        if not os.path.exists(image_directory):
            self.status_label.setText("Images directory not found")
            return
        if not os.path.exists(mask_directory):
            self.status_label.setText("Masks directory not found")
            return
        
        # Show progress dialog
        self.progress_dialog = QProgressDialog("Extracting contrast data...", None, 0, 0, self)
        self.progress_dialog.setWindowTitle("Loading")
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setMinimumDuration(0)
        self.progress_dialog.setCancelButton(None)
        self.progress_dialog.show()
        QApplication.processEvents()
        
        # Start loader thread
        use_flatfield = os.path.exists(flatfield_path)
        self.loader_thread = ContrastLoaderThread(
            image_directory, mask_directory, flatfield_path, use_flatfield
        )
        self.loader_thread.progress_update.connect(self.on_loading_progress)
        self.loader_thread.finished.connect(self.on_loading_finished)
        self.loader_thread.error.connect(self.on_loading_error)
        self.loader_thread.start()
    
    def on_loading_progress(self, message):
        """Update progress dialog with message"""
        if self.progress_dialog:
            self.progress_dialog.setLabelText(message)
            QApplication.processEvents()
    
    def on_loading_finished(self, instance_contrasts, instance_classifiers):
        """Handle completed contrast data loading"""
        if self.progress_dialog:
            self.progress_dialog.close()
            self.progress_dialog = None
        
        if instance_contrasts and len(instance_contrasts) > 0:
            # Apply high standard deviation filter using UI threshold
            threshold = self.stddev_threshold_input.value()
            filtered_contrasts, filtered_classifiers = self.apply_high_stddev_filter(
                instance_contrasts, instance_classifiers, threshold=threshold
            )
            
            # Convert list of arrays to single array with mean contrast per instance
            # Note: contrast data is in BGR format, convert to RGB for display
            contrast_bgr = np.array([np.mean(instance_contrast, axis=0) 
                                   for instance_contrast in filtered_contrasts])
            # Convert BGR to RGB: [B, G, R] -> [R, G, B]
            self.contrast_data = contrast_bgr[:, [2, 1, 0]]  # Swap B and R channels
            self.instance_classifiers = filtered_classifiers
            
            # Initialize class assignments (all start as unassigned = -1)
            self.class_assignments = np.full(len(self.contrast_data), -1, dtype=int)
            
            # Update UI with filtering info
            original_count = len(instance_contrasts)
            filtered_count = len(filtered_contrasts)
            filtered_out = original_count - filtered_count
            
            self.progress_label.setText(f"Loaded {filtered_count} instances ({filtered_out} filtered)")
            self.btn_save.setEnabled(True)
            
            # Create plots
            self.update_plots()
            
            self.status_label.setText(f"Ready - {filtered_count} instances loaded, {filtered_out} high-stddev filtered")
        else:
            self.status_label.setText("No contrast data found")
    
    def on_loading_error(self, error_msg):
        """Handle loading error"""
        if self.progress_dialog:
            self.progress_dialog.close()
            self.progress_dialog = None
        
        self.status_label.setText(f"Error loading data: {error_msg}")
        QMessageBox.warning(self, "Error", f"Failed to load contrast data: {error_msg}")
    
    def apply_high_stddev_filter(self, instance_contrasts, instance_classifiers, threshold=0.2):
        """
        Filter out instances with high standard deviation in contrast values.
        Similar to the implementation in zConvert_instance_masks_to_semantic_masks.ipynb
        
        Args:
            instance_contrasts: List of contrast arrays for each instance
            instance_classifiers: List of (mask_name, instance_id) tuples
            threshold: Standard deviation threshold (default 0.2)
            
        Returns:
            Tuple of (filtered_contrasts, filtered_classifiers)
        """
        filtered_contrasts = []
        filtered_classifiers = []
        
        total_instances = len(instance_contrasts)
        filtered_count = 0
        
        for i, instance_contrast in enumerate(instance_contrasts):
            # Calculate standard deviation for each channel (BGR)
            std_b = np.std(instance_contrast[:, 0])
            std_g = np.std(instance_contrast[:, 1]) 
            std_r = np.std(instance_contrast[:, 2])
            
            # Calculate mean standard deviation across all channels
            mean_std = np.mean([std_r, std_g, std_b])
            
            # Keep instance if standard deviation is below threshold
            if mean_std <= threshold:
                filtered_contrasts.append(instance_contrast)
                filtered_classifiers.append(instance_classifiers[i])
            else:
                filtered_count += 1
        
        print(f"High-stddev filter: Removed {filtered_count}/{total_instances} instances (threshold={threshold})")
        
        return filtered_contrasts, filtered_classifiers
    
    def update_plots(self):
        """Update all three scatter plots with current data and class assignments"""
        if self.contrast_data is None:
            return
            
        # Extract RGB contrast values
        r_values = self.contrast_data[:, 0]  # Red contrast
        g_values = self.contrast_data[:, 1]  # Green contrast  
        b_values = self.contrast_data[:, 2]  # Blue contrast
        
        # Create color map for class assignments
        colors = self.get_class_colors()
        
        # Clear and update RG plot (Green vs Red)
        self.ax_rg.clear()
        # Create heatmap background
        self.create_heatmap_background(self.ax_rg, g_values, r_values, self.g_limits, self.r_limits)
        # Scatter plot on top with larger points
        scatter_rg = self.ax_rg.scatter(g_values, r_values, c=colors, alpha=0.8, s=35, edgecolors='black', linewidth=0.5)
        self.ax_rg.set_xlabel('Green Contrast')
        self.ax_rg.set_ylabel('Red Contrast')
        self.ax_rg.set_title('Green vs Red')
        self.ax_rg.grid(True, alpha=0.3)
        self.ax_rg.set_xlim(self.g_limits)
        self.ax_rg.set_ylim(self.r_limits)
        
        # Clear and update GB plot (Blue vs Green)
        self.ax_gb.clear()
        # Create heatmap background
        self.create_heatmap_background(self.ax_gb, b_values, g_values, self.b_limits, self.g_limits)
        # Scatter plot on top with larger points
        scatter_gb = self.ax_gb.scatter(b_values, g_values, c=colors, alpha=0.8, s=35, edgecolors='black', linewidth=0.5)
        self.ax_gb.set_xlabel('Blue Contrast')
        self.ax_gb.set_ylabel('Green Contrast')
        self.ax_gb.set_title('Blue vs Green')
        self.ax_gb.grid(True, alpha=0.3)
        self.ax_gb.set_xlim(self.b_limits)
        self.ax_gb.set_ylim(self.g_limits)
        
        # Clear and update BR plot (Red vs Blue)
        self.ax_br.clear()
        # Create heatmap background
        self.create_heatmap_background(self.ax_br, r_values, b_values, self.r_limits, self.b_limits)
        # Scatter plot on top with larger points
        scatter_br = self.ax_br.scatter(r_values, b_values, c=colors, alpha=0.8, s=35, edgecolors='black', linewidth=0.5)
        self.ax_br.set_xlabel('Red Contrast')
        self.ax_br.set_ylabel('Blue Contrast') 
        self.ax_br.set_title('Red vs Blue')
        self.ax_br.grid(True, alpha=0.3)
        self.ax_br.set_xlim(self.r_limits)
        self.ax_br.set_ylim(self.b_limits)
        
        # Set up lasso selectors
        self.setup_lasso_selectors(r_values, g_values, b_values)
        
        # Refresh canvases
        self.canvas_rg.draw()
        self.canvas_gb.draw()
        self.canvas_br.draw()
        
    def create_heatmap_background(self, ax, x_values, y_values, x_range, y_range, bins=50):
        """Create a 2D histogram heatmap as background"""
        # Create 2D histogram
        hist, xedges, yedges = np.histogram2d(
            x_values, y_values, 
            bins=bins, 
            range=[x_range, y_range]
        )
        
        # Apply gaussian filter for smoothing
        hist_smooth = gaussian_filter(hist, sigma=1.5)
        
        # Log transform for better visualization
        hist_log = np.log(hist_smooth + 1)
        
        # Display as heatmap
        extent = [x_range[0], x_range[1], y_range[0], y_range[1]]
        ax.imshow(hist_log.T, origin='lower', extent=extent, 
                 cmap='plasma', alpha=0.3, aspect='auto')
        
    def get_class_colors(self):
        """Get colors for each point based on class assignment"""
        colors = []
        for assignment in self.class_assignments:
            if assignment == -1:
                colors.append('gray')  # Unassigned
            elif assignment == 0:
                colors.append('black')  # Background
            else:
                # Use a colormap for different classes
                cmap = plt.cm.tab10
                colors.append(cmap(assignment % 10))
        return colors
        
    def setup_lasso_selectors(self, r_values, g_values, b_values):
        """Set up lasso selection tools for all three plots"""
        # RG plot lasso (Green vs Red - X=Green, Y=Red)
        def onselect_rg(verts):
            self.on_lasso_select(verts, g_values, r_values)
            
        self.lasso_rg = LassoSelector(self.ax_rg, onselect_rg, useblit=True)
        
        # GB plot lasso (Blue vs Green - X=Blue, Y=Green)
        def onselect_gb(verts):
            self.on_lasso_select(verts, b_values, g_values)
            
        self.lasso_gb = LassoSelector(self.ax_gb, onselect_gb, useblit=True)
        
        # BR plot lasso (Red vs Blue - X=Red, Y=Blue)
        def onselect_br(verts):
            self.on_lasso_select(verts, r_values, b_values)
            
        self.lasso_br = LassoSelector(self.ax_br, onselect_br, useblit=True)
        
    def on_lasso_select(self, verts, x_values, y_values):
        """Handle lasso selection and assign current class to selected points"""
        if len(verts) < 3:  # Need at least 3 points to form a polygon
            return
            
        # Create path from lasso vertices
        path = Path(verts)
        
        # Find points inside the lasso
        points = np.column_stack([x_values, y_values])
        inside = path.contains_points(points)
        
        # Assign current class to selected points
        self.class_assignments[inside] = self.current_class
        
        # Update plots to show new assignments
        self.update_plots()
        
        # Update progress
        assigned_count = np.sum(self.class_assignments != -1)
        total_count = len(self.class_assignments)
        self.progress_label.setText(f"Assigned: {assigned_count}/{total_count}")
        
    def on_class_changed(self, value):
        """Handle class selection change"""
        self.current_class = value
        
    def on_r_limits_changed(self):
        """Handle red axis limits change"""
        self.r_limits = [self.r_min_spinbox.value(), self.r_max_spinbox.value()]
        self.update_plots()
        
    def on_g_limits_changed(self):
        """Handle green axis limits change"""
        self.g_limits = [self.g_min_spinbox.value(), self.g_max_spinbox.value()]
        self.update_plots()
        
    def on_b_limits_changed(self):
        """Handle blue axis limits change"""
        self.b_limits = [self.b_min_spinbox.value(), self.b_max_spinbox.value()]
        self.update_plots()
        
    def save_semantic_masks(self):
        """Save semantic masks based on class assignments"""
        if self.contrast_data is None or self.class_assignments is None:
            QMessageBox.warning(self, "Error", "No data to save")
            return
            
        if not hasattr(self, 'instance_classifiers'):
            QMessageBox.warning(self, "Error", "No instance classifier data available")
            return
            
        try:
            # Create semantic masks directory
            semantic_dir = os.path.join(self.project_folder, "semantic_masks")
            os.makedirs(semantic_dir, exist_ok=True)
            
            # Group instance classifiers by mask name
            mask_instances = {}
            for i, (mask_name, instance_id) in enumerate(self.instance_classifiers):
                if mask_name not in mask_instances:
                    mask_instances[mask_name] = []
                mask_instances[mask_name].append((instance_id, self.class_assignments[i]))
            
            saved_count = 0
            mask_directory = os.path.join(self.project_folder, "masks")
            
            # Process each mask file
            for mask_name, instances in mask_instances.items():
                mask_path = os.path.join(mask_directory, mask_name)
                
                if not os.path.exists(mask_path):
                    continue
                    
                # Load instance mask
                instance_mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
                if instance_mask is None:
                    continue
                    
                # Create semantic mask (same size, but class values instead of instance IDs)
                semantic_mask = np.zeros_like(instance_mask, dtype=np.uint8)
                
                # Map each instance to its assigned class
                for instance_id, class_assignment in instances:
                    if class_assignment != -1:  # Only process assigned instances
                        # Find pixels belonging to this instance
                        instance_pixels = (instance_mask == instance_id)
                        # Set them to the assigned class
                        semantic_mask[instance_pixels] = class_assignment
                
                # Save semantic mask with same name as mask (same name as original image)
                semantic_path = os.path.join(semantic_dir, mask_name)
                cv2.imwrite(semantic_path, semantic_mask)
                saved_count += 1
            
            # Also save class assignment mapping for future reference
            assignments_path = os.path.join(semantic_dir, "class_assignments.npz")
            np.savez(assignments_path, 
                    class_assignments=self.class_assignments,
                    instance_classifiers=np.array(self.instance_classifiers, dtype=object))
            
            # Generate statistics
            stats = self.generate_class_statistics()
            stats_path = os.path.join(semantic_dir, "class_statistics.txt")
            with open(stats_path, 'w') as f:
                f.write(stats)
            
            QMessageBox.information(self, "Success", 
                                  f"Semantic masks saved successfully!\n"
                                  f"Saved {saved_count} semantic masks to:\n{semantic_dir}")
            
            self.status_label.setText(f"Saved {saved_count} semantic masks")
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to save semantic masks: {e}")
            import traceback
            traceback.print_exc()
            
    def generate_class_statistics(self):
        """Generate statistics about class assignments"""
        if self.class_assignments is None:
            return "No data available"
            
        stats = ["Class Assignment Statistics", "=" * 30, ""]
        
        # Count instances per class
        unique_classes, counts = np.unique(self.class_assignments, return_counts=True)
        
        total_instances = len(self.class_assignments)
        assigned_instances = np.sum(self.class_assignments != -1)
        unassigned_instances = total_instances - assigned_instances
        
        stats.append(f"Total instances: {total_instances}")
        stats.append(f"Assigned instances: {assigned_instances}")
        stats.append(f"Unassigned instances: {unassigned_instances}")
        stats.append(f"Assignment rate: {assigned_instances/total_instances*100:.1f}%")
        stats.append("")
        
        stats.append("Class distribution:")
        for class_id, count in zip(unique_classes, counts):
            if class_id == -1:
                stats.append(f"  Unassigned: {count} instances")
            else:
                percentage = count / total_instances * 100
                stats.append(f"  Class {class_id}: {count} instances ({percentage:.1f}%)")
        
        return "\n".join(stats)
            
    def handle_key_press(self, key):
        """Handle keyboard shortcuts - centralized method"""
        
        # Update shortcuts label to show activity
        self.shortcuts_label.setStyleSheet("color: blue; font-size: 10px; font-weight: bold;")
        
        if key == Qt.Key_D:
            # Increment class
            new_value = min(self.class_spinbox.value() + 1, 10)
            self.class_spinbox.setValue(new_value)
            self.status_label.setText(f"[D] Class incremented to {new_value}")
        elif key == Qt.Key_A:
            # Decrement class  
            new_value = max(self.class_spinbox.value() - 1, 0)
            self.class_spinbox.setValue(new_value)
            self.status_label.setText(f"[A] Class decremented to {new_value}")
        elif key == Qt.Key_X:
            # Set to background class
            self.class_spinbox.setValue(0)
            self.status_label.setText("[X] Set to background class (0)")
        elif key == Qt.Key_S:
            # Save semantic masks
            self.status_label.setText("[S] Saving semantic masks...")
            self.save_semantic_masks()
        elif key == Qt.Key_C:
            # Clear all class assignments
            self.status_label.setText("[C] Clearing all assignments...")
            self.clear_all_assignments()
        elif key == Qt.Key_Space:
            # Toggle between showing all points vs only unassigned
            self.status_label.setText("[Space] Toggling view mode...")
            self.toggle_view_mode()
        elif key >= Qt.Key_1 and key <= Qt.Key_9:
            # Quick class selection (1-9)
            class_num = key - Qt.Key_0
            self.class_spinbox.setValue(class_num)
            self.status_label.setText(f"[{class_num}] Quick select class {class_num}")
        elif key == Qt.Key_0:
            # Quick background class
            self.class_spinbox.setValue(0)
            self.status_label.setText("[0] Quick select background class (0)")
        elif key == Qt.Key_Escape:
            # Close dialog
            self.status_label.setText("[Esc] Closing dialog...")
            self.close()
        else:
            self.status_label.setText(f"Unknown key pressed: {key}")
            
        # Reset shortcuts label color after a moment
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(1000, lambda: self.shortcuts_label.setStyleSheet("color: gray; font-size: 10px;"))
    
    def keyPressEvent(self, event):
        """Handle keyboard shortcuts"""
        self.handle_key_press(event.key())
            
    def clear_all_assignments(self):
        """Clear all class assignments"""
        if self.class_assignments is not None:
            self.class_assignments.fill(-1)  # Reset all to unassigned
            self.update_plots()
            self.progress_label.setText(f"Assigned: 0/{len(self.class_assignments)}")
            self.status_label.setText("All assignments cleared")
            
    def toggle_view_mode(self):
        """Toggle between showing all points vs highlighting unassigned"""
        # This could be expanded to show different visualization modes
        self.update_plots()
        self.status_label.setText("View toggled")
        
    def showEvent(self, event):
        """Ensure dialog gets focus when shown and is maximized"""
        super().showEvent(event)
        self.setWindowState(Qt.WindowMaximized)
        self.setFocus()
        self.activateWindow()
        self.raise_()
        
    def mousePressEvent(self, event):
        """Regain focus when dialog is clicked"""
        super().mousePressEvent(event)
        self.setFocus()
    
    def show_help(self):
        """Show help dialog for the class annotator window"""
        show_help_dialog(self, "class_annotator_dialog", "Class Annotator Help")
    
    def keyPressEvent(self, event):
        """Handle keyboard shortcuts and class selection"""
        key = event.key()
        
        # F1 - Show help
        if key == Qt.Key_F1:
            self.show_help()
        
        # Number keys 0-9 for class selection
        elif Qt.Key_0 <= key <= Qt.Key_9:
            class_number = key - Qt.Key_0
            if class_number <= 10:  # Maximum classes allowed
                self.class_spinbox.setValue(class_number)
                self.on_class_changed(class_number)
        
        else:
            super().keyPressEvent(event)