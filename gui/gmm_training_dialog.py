import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.patches import Circle
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QLineEdit, QSpinBox, QDoubleSpinBox, 
                             QGroupBox, QGridLayout, QTextEdit, 
                             QProgressBar, QMessageBox, QCheckBox, QComboBox,
                             QSplitter, QWidget, QScrollArea, QProgressDialog,
                             QApplication)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from gui.help_dialog import show_help_dialog


# Color palette for different components (contrasting colors)
MARKER_COLORS = [
    ('#00FFFF', '#FFD700'),  # Cyan with gold edge
    ('#FF1493', '#00FF00'),  # Deep pink with green edge
    ('#FF4500', '#00CED1'),  # Orange-red with dark turquoise edge
    ('#9370DB', '#FFFF00'),  # Medium purple with yellow edge
    ('#32CD32', '#FF00FF'),  # Lime green with magenta edge
    ('#FF69B4', '#00FFFF'),  # Hot pink with cyan edge
    ('#FFD700', '#FF1493'),  # Gold with deep pink edge
    ('#00CED1', '#FF4500'),  # Dark turquoise with orange-red edge
    ('#FFFF00', '#9370DB'),  # Yellow with medium purple edge
    ('#FF00FF', '#32CD32'),  # Magenta with lime green edge
    ('#00FF00', '#FF1493'),  # Green with deep pink edge
    ('#1E90FF', '#FFD700'),  # Dodger blue with gold edge
    ('#FF6347', '#00FFFF'),  # Tomato with cyan edge
    ('#7FFF00', '#FF00FF'),  # Chartreuse with magenta edge
    ('#FF8C00', '#00CED1'),  # Dark orange with dark turquoise edge
]


class GMMTrainingThread(QThread):
    """Separate thread for GMM training based on zInteractive_Parameter_Estimation.ipynb"""
    progress_update = pyqtSignal(int)
    log_message = pyqtSignal(str)
    training_complete = pyqtSignal(bool)
    plot_gaussians = pyqtSignal(object, object)  # means, covariances
    
    def __init__(self, image_dir, mask_dir, save_dir, params, flatfield_path=None, initial_means=None, class_names=None, noise_flags=None):
        super().__init__()
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.save_dir = save_dir
        self.params = params
        self.flatfield_path = flatfield_path
        self.initial_means = initial_means
        self.class_names = class_names
        self.noise_flags = noise_flags  # List of booleans indicating which components are noise
        self.should_stop = False
        
    def run(self):
        """Run GMM training exactly like the interactive notebook"""
        try:
            self.log_message.emit("Starting GMM training...")
            
            # Add scripts to path
            scripts_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts")
            if scripts_path not in sys.path:
                sys.path.append(scripts_path)
            
            from scripts.preprocessor_functions import get_contrasts_from_dir
            from scripts.plotting_functions import create_heatmap_plot
            from scripts.fitting_functions import fit_set
            from scripts.postprocessing_functions import format_components
            
            self.progress_update.emit(10)
            self.log_message.emit("Loading contrast values from images and masks...")
            
            # Load contrast data - function returns (full_colors, full_contrasts, background_colors)
            _, datapoints_contrast, _ = get_contrasts_from_dir(
                image_directory=self.image_dir,
                mask_directory=self.mask_dir,
                flatfield_path=self.flatfield_path,
            )
            
            self.progress_update.emit(20)
            self.log_message.emit(f"Loaded {len(datapoints_contrast)} contrast datapoints")
            
            if self.should_stop:
                return
            
            # Always crop the contrast array using the bounds
            upper_bounds = self.params['upper_bounds']
            lower_bounds = self.params['lower_bounds']
            
            datapoints_contrast_cropped = datapoints_contrast[
                (datapoints_contrast[:, 0] > lower_bounds[0])
                & (datapoints_contrast[:, 0] < upper_bounds[0])
                & (datapoints_contrast[:, 1] > lower_bounds[1])
                & (datapoints_contrast[:, 1] < upper_bounds[1])
                & (datapoints_contrast[:, 2] > lower_bounds[2])
                & (datapoints_contrast[:, 2] < upper_bounds[2])
            ]
            self.log_message.emit(f"Cropped to {len(datapoints_contrast_cropped)} datapoints")
            
            self.progress_update.emit(30)
            
            if self.should_stop:
                return
            
            # Fit the GMM
            self.log_message.emit("Fitting Gaussian Mixture Model...")
            
            # Prepare initial means - user can provide partial initial means
            initial_means_for_fit = None
            
            if self.initial_means is not None:
                num_user_means = len(self.initial_means)
                total_components = self.params['num_components']
                
                self.log_message.emit(f"User provided {num_user_means} initial means for {total_components} total components")
                
                # Start with user-provided means
                means_list = [self.initial_means]
                
                # Add random means for remaining components (if any)
                num_remaining = total_components - num_user_means
                if num_remaining > 0:
                    self.log_message.emit(f"Generating {num_remaining} random initial means for remaining components")
                    random_means = np.random.uniform(
                        low=[lower_bounds[0], lower_bounds[1], lower_bounds[2]],
                        high=[upper_bounds[0], upper_bounds[1], upper_bounds[2]],
                        size=(num_remaining, 3)
                    )
                    means_list.append(random_means)
                
                # Combine all means
                initial_means_for_fit = np.vstack(means_list)
                self.log_message.emit(f"Total initial means: {len(initial_means_for_fit)} ({num_user_means} user-provided + {num_remaining} randomly generated)")
            
            (
                all_means_gauss,
                all_covariances_gauss,
                all_weights_gauss,
                sampled_data,
                predicted_labels,
            ) = fit_set(
                data=datapoints_contrast_cropped,
                num_components=self.params['num_components'],
                num_additional_noise_comp=self.params['num_noise_components'],
                cov_type=self.params['cov_type'],
                sample_size=self.params['sample_size'],
                used_channels=self.params['used_channels'],
                initial_means=initial_means_for_fit,
            )
            
            self.progress_update.emit(80)
            self.log_message.emit("GMM fitting completed!")
            
            if self.should_stop:
                return
            
            # Remove user-marked noise components if any
            if self.noise_flags and any(self.noise_flags):
                noise_indices = [i for i, is_noise in enumerate(self.noise_flags) if is_noise]
                self.log_message.emit(f"Removing {len(noise_indices)} user-marked noise component(s)...")
                
                # Get indices of components to keep (non-noise user components + any randomly initialized)
                num_user_components = len(self.noise_flags)
                total_fitted = len(all_means_gauss)
                
                # Build list of indices to keep
                keep_indices = []
                removed_names = []
                
                for i in range(total_fitted):
                    if i < num_user_components:
                        # This is a user-specified component
                        if not self.noise_flags[i]:
                            keep_indices.append(i)
                        else:
                            if self.class_names and i < len(self.class_names):
                                removed_names.append(self.class_names[i])
                    else:
                        # This is a randomly initialized component, keep it
                        keep_indices.append(i)
                
                if removed_names:
                    self.log_message.emit(f"Removed components: {', '.join(removed_names)}")
                
                # Filter arrays
                all_means_gauss = all_means_gauss[keep_indices]
                all_covariances_gauss = all_covariances_gauss[keep_indices]
                
                # Filter class names
                if self.class_names:
                    self.class_names = [self.class_names[i] for i in range(len(self.class_names)) if i < num_user_components and not self.noise_flags[i]]
                
                self.log_message.emit(f"Final model has {len(all_means_gauss)} component(s)")
            
            # Format and save the components
            self.log_message.emit("Saving GMM parameters...")
            os.makedirs(self.save_dir, exist_ok=True)
            
            component_dict = format_components(all_means_gauss, all_covariances_gauss, self.class_names)
            
            output_path = os.path.join(self.save_dir, "GMM_parameters.json")
            with open(output_path, "w") as f:
                json.dump(component_dict, f, indent=4, sort_keys=True)
            
            # Emit signal to plot Gaussians on the main thread
            self.plot_gaussians.emit(all_means_gauss, all_covariances_gauss)
            
            self.progress_update.emit(100)
            self.log_message.emit(f"Training completed! GMM parameters saved to: {output_path}")
            self.training_complete.emit(True)
            
        except Exception as e:
            self.log_message.emit(f"Error: {e}")
            import traceback
            traceback.print_exc()
            self.training_complete.emit(False)
    
    def stop(self):
        """Stop the training thread"""
        self.should_stop = True


class GMMContrastLoaderThread(QThread):
    """Thread for loading contrast data without blocking the UI"""
    progress_update = pyqtSignal(str)
    finished = pyqtSignal(object)  # contrast_data
    error = pyqtSignal(str)
    
    def __init__(self, image_dir, mask_dir, flatfield_path):
        super().__init__()
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.flatfield_path = flatfield_path
        
    def run(self):
        try:
            self.progress_update.emit("Extracting contrast data from images...")
            
            # Add scripts to path
            scripts_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts")
            if scripts_path not in sys.path:
                sys.path.append(scripts_path)
            
            from scripts.preprocessor_functions import get_contrasts_from_dir
            
            # Load contrast data
            _, contrast_data, _ = get_contrasts_from_dir(
                image_directory=self.image_dir,
                mask_directory=self.mask_dir,
                flatfield_path=self.flatfield_path,
            )
            
            self.finished.emit(contrast_data)
            
        except Exception as e:
            self.error.emit(str(e))


class GMMTrainingDialog(QDialog):
    """Dialog for training GMM classifier based on zInteractive_Parameter_Estimation.ipynb"""
    
    def __init__(self, parent=None, material_folder=None):
        super().__init__(parent)
        self.setWindowTitle("GMM Training Dashboard")
        
        # Make dialog full screen
        if parent:
            self.setWindowState(Qt.WindowMaximized)
        else:
            self.showMaximized()
        
        self.training_thread = None
        self.loader_thread = None
        self.progress_dialog = None
        self.material_folder = material_folder
        self.contrast_data = None
        self.canvas = None
        self.gmm_means = None  # Store GMM means for plotting
        self.gmm_covariances = None  # Store GMM covariances for plotting
        self.draggable_markers = []  # Store draggable marker objects for cleanup
        self.currently_dragging = None  # Track which marker is being dragged
        
        self.init_ui()
        
        # Auto-populate paths if material folder is provided
        if material_folder:
            self.auto_populate_paths(material_folder)
            # Automatically load and plot contrast data
            self.load_and_plot_data()
    
    def init_ui(self):
        """Initialize the user interface"""
        main_layout = QVBoxLayout()
        
        # Create a horizontal splitter for left (controls) and right sections
        h_splitter = QSplitter(Qt.Horizontal)
        
        # Left side - Controls (1/5 of width)
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        
        # Header with help button
        header_layout = QHBoxLayout()
        header_layout.addWidget(QLabel("<h2>GMM Training</h2>"))
        header_layout.addStretch()
        
        help_btn = QPushButton("Help")
        help_btn.setStyleSheet("""
            QPushButton {
                background-color: #0078D4;
                color: white;
                border-radius: 3px;
                padding: 5px 15px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #106EBE;
            }
        """)
        help_btn.clicked.connect(self.show_help)
        header_layout.addWidget(help_btn)
        
        left_layout.addLayout(header_layout)
        
        # Data Configuration
        data_group = QGroupBox("Data Configuration")
        data_layout = QGridLayout()
        
        data_layout.addWidget(QLabel("Images Directory:"), 0, 0)
        self.image_dir_edit = QLineEdit()
        self.image_dir_edit.setPlaceholderText("Auto-populated from material folder")
        self.image_dir_edit.setReadOnly(True)
        data_layout.addWidget(self.image_dir_edit, 0, 1)
        
        data_layout.addWidget(QLabel("Masks Directory:"), 1, 0)
        self.mask_dir_edit = QLineEdit()
        self.mask_dir_edit.setPlaceholderText("Auto-populated from material folder")
        self.mask_dir_edit.setReadOnly(True)
        data_layout.addWidget(self.mask_dir_edit, 1, 1)
        
        data_layout.addWidget(QLabel("Flatfield Image:"), 2, 0)
        self.flatfield_edit = QLineEdit()
        self.flatfield_edit.setPlaceholderText("Auto-populated from material folder")
        self.flatfield_edit.setReadOnly(True)
        data_layout.addWidget(self.flatfield_edit, 2, 1)
        
        data_layout.addWidget(QLabel("Save Directory:"), 3, 0)
        self.save_dir_edit = QLineEdit()
        self.save_dir_edit.setPlaceholderText("Auto-populated from material folder")
        self.save_dir_edit.setReadOnly(True)
        data_layout.addWidget(self.save_dir_edit, 3, 1)
        
        data_group.setLayout(data_layout)
        left_layout.addWidget(data_group)
        
        # GMM Parameters
        gmm_group = QGroupBox("GMM Parameters")
        gmm_layout = QGridLayout()
        
        gmm_layout.addWidget(QLabel("Number of Components:"), 0, 0)
        self.num_components_spin = QSpinBox()
        self.num_components_spin.setRange(1, 20)
        self.num_components_spin.setValue(9)
        gmm_layout.addWidget(self.num_components_spin, 0, 1)
        
        gmm_layout.addWidget(QLabel("Noise Components:"), 0, 2)
        self.num_noise_spin = QSpinBox()
        self.num_noise_spin.setRange(0, 10)
        self.num_noise_spin.setValue(0)
        gmm_layout.addWidget(self.num_noise_spin, 0, 3)
        
        gmm_layout.addWidget(QLabel("Sample Size:"), 1, 0)
        self.sample_size_spin = QSpinBox()
        self.sample_size_spin.setRange(1000, 100000)
        self.sample_size_spin.setValue(45000)
        self.sample_size_spin.setSingleStep(5000)
        gmm_layout.addWidget(self.sample_size_spin, 1, 1)
        
        gmm_layout.addWidget(QLabel("Covariance Type:"), 1, 2)
        self.cov_type_combo = QComboBox()
        self.cov_type_combo.addItems(["full", "tied", "diag"])
        gmm_layout.addWidget(self.cov_type_combo, 1, 3)
        
        gmm_layout.addWidget(QLabel("Used Channels:"), 2, 0)
        self.used_channels_combo = QComboBox()
        self.used_channels_combo.addItems(["BGR", "BG", "GR", "BR"])
        self.used_channels_combo.currentTextChanged.connect(self.on_channels_changed)
        gmm_layout.addWidget(self.used_channels_combo, 2, 1)
        
        gmm_group.setLayout(gmm_layout)
        left_layout.addWidget(gmm_group)
        
        # Data Cropping (optional)
        crop_group = QGroupBox("Data Cropping (Optional)")
        crop_layout = QGridLayout()
        
        # Auto-crop controls
        auto_crop_layout = QHBoxLayout()
        self.btn_auto_crop = QPushButton("Auto-Crop")
        self.btn_auto_crop.setToolTip("Automatically set bounds based on data range ± padding %")
        self.btn_auto_crop.clicked.connect(self.auto_crop_bounds)
        self.btn_auto_crop.setStyleSheet("""
            QPushButton {
                background-color: #0078D4;
                color: white;
                border-radius: 3px;
                padding: 3px 8px;
            }
            QPushButton:hover {
                background-color: #106EBE;
            }
        """)
        auto_crop_layout.addWidget(self.btn_auto_crop)
        
        auto_crop_layout.addWidget(QLabel("Padding %:"))
        self.auto_crop_padding_spin = QSpinBox()
        self.auto_crop_padding_spin.setRange(0, 50)
        self.auto_crop_padding_spin.setValue(5)
        self.auto_crop_padding_spin.setToolTip("Percentage padding to add beyond min/max values")
        auto_crop_layout.addWidget(self.auto_crop_padding_spin)
        auto_crop_layout.addStretch()
        crop_layout.addLayout(auto_crop_layout, 0, 0, 1, 4)
        
        crop_layout.addWidget(QLabel("Lower Bounds (B,G,R):"), 1, 0)
        self.lower_b_spin = QDoubleSpinBox()
        self.lower_b_spin.setRange(-5.0, 5.0)
        self.lower_b_spin.setValue(-1.0)
        self.lower_b_spin.setSingleStep(0.1)
        self.lower_b_spin.setDecimals(2)
        self.lower_b_spin.valueChanged.connect(self.on_bounds_changed)
        crop_layout.addWidget(self.lower_b_spin, 1, 1)
        
        self.lower_g_spin = QDoubleSpinBox()
        self.lower_g_spin.setRange(-5.0, 5.0)
        self.lower_g_spin.setValue(-1.5)
        self.lower_g_spin.setSingleStep(0.1)
        self.lower_g_spin.setDecimals(2)
        self.lower_g_spin.valueChanged.connect(self.on_bounds_changed)
        crop_layout.addWidget(self.lower_g_spin, 1, 2)
        
        self.lower_r_spin = QDoubleSpinBox()
        self.lower_r_spin.setRange(-5.0, 5.0)
        self.lower_r_spin.setValue(-1.5)
        self.lower_r_spin.setSingleStep(0.1)
        self.lower_r_spin.setDecimals(2)
        self.lower_r_spin.valueChanged.connect(self.on_bounds_changed)
        crop_layout.addWidget(self.lower_r_spin, 1, 3)
        
        crop_layout.addWidget(QLabel("Upper Bounds (B,G,R):"), 2, 0)
        self.upper_b_spin = QDoubleSpinBox()
        self.upper_b_spin.setRange(-5.0, 5.0)
        self.upper_b_spin.setValue(0.5)
        self.upper_b_spin.setSingleStep(0.1)
        self.upper_b_spin.setDecimals(2)
        self.upper_b_spin.valueChanged.connect(self.on_bounds_changed)
        crop_layout.addWidget(self.upper_b_spin, 2, 1)
        
        self.upper_g_spin = QDoubleSpinBox()
        self.upper_g_spin.setRange(-5.0, 5.0)
        self.upper_g_spin.setValue(0.5)
        self.upper_g_spin.setSingleStep(0.1)
        self.upper_g_spin.setDecimals(2)
        self.upper_g_spin.valueChanged.connect(self.on_bounds_changed)
        crop_layout.addWidget(self.upper_g_spin, 2, 2)
        
        self.upper_r_spin = QDoubleSpinBox()
        self.upper_r_spin.setRange(-5.0, 5.0)
        self.upper_r_spin.setValue(0.5)
        self.upper_r_spin.setSingleStep(0.1)
        self.upper_r_spin.setDecimals(2)
        self.upper_r_spin.valueChanged.connect(self.on_bounds_changed)
        crop_layout.addWidget(self.upper_r_spin, 2, 3)
        
        crop_group.setLayout(crop_layout)
        left_layout.addWidget(crop_group)
        
        # Refresh plot button
        refresh_layout = QHBoxLayout()
        self.btn_refresh_plot = QPushButton("Refresh Plot")
        self.btn_refresh_plot.clicked.connect(self.refresh_plot)
        self.btn_refresh_plot.setStyleSheet("""
            QPushButton {
                background-color: #107C10;
                color: white;
                border-radius: 3px;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: #0E6B0E;
            }
        """)
        refresh_layout.addWidget(self.btn_refresh_plot)
        refresh_layout.addStretch()
        left_layout.addLayout(refresh_layout)
        
        # Progress bar
        progress_layout = QHBoxLayout()
        progress_layout.addWidget(QLabel("Progress:"))
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        progress_layout.addWidget(self.progress_bar)
        left_layout.addLayout(progress_layout)
        
        # Control buttons
        buttons_layout = QHBoxLayout()
        
        self.btn_start = QPushButton("Start Training")
        self.btn_start.clicked.connect(self.start_training)
        buttons_layout.addWidget(self.btn_start)
        
        self.btn_stop = QPushButton("Stop Training")
        self.btn_stop.clicked.connect(self.stop_training)
        self.btn_stop.setEnabled(False)
        buttons_layout.addWidget(self.btn_stop)
        
        self.btn_close = QPushButton("Close")
        self.btn_close.clicked.connect(self.close)
        buttons_layout.addWidget(self.btn_close)
        
        left_layout.addLayout(buttons_layout)
        
        # Training log
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(150)
        self.log_text.setReadOnly(True)
        left_layout.addWidget(QLabel("Training Log:"))
        left_layout.addWidget(self.log_text)
        
        left_widget.setLayout(left_layout)
        
        # Right side - Vertical splitter for plots (top) and initial means (bottom)
        right_widget = QWidget()
        right_main_layout = QVBoxLayout()
        
        v_splitter = QSplitter(Qt.Vertical)
        
        # Top section - Plots
        plot_widget = QWidget()
        plot_layout = QVBoxLayout()
        plot_layout.addWidget(QLabel("<h3>Contrast Distribution Heatmap</h3>"))
        
        # Matplotlib canvas
        self.figure = plt.figure(figsize=(16, 6), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumSize(800, 400)
        
        # Connect centralized event handlers for dragging
        self.canvas.mpl_connect('button_press_event', self.on_canvas_press)
        self.canvas.mpl_connect('button_release_event', self.on_canvas_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_canvas_motion)
        
        plot_layout.addWidget(self.canvas)
        plot_widget.setLayout(plot_layout)
        
        # Bottom section - Initial Means Configuration
        initial_means_widget = QWidget()
        initial_means_main_layout = QVBoxLayout()
        
        initial_means_group = QGroupBox("Initial Means (Optional)")
        initial_means_layout = QVBoxLayout()
        
        # Header for initial means
        im_header_layout = QHBoxLayout()
        im_header_layout.addWidget(QLabel("<b>Click on plot to add mean, or:</b>"))
        im_header_layout.addStretch()
        self.btn_add_mean = QPushButton("Add Component at (0,0,0)")
        self.btn_add_mean.clicked.connect(self.add_initial_mean_row)
        im_header_layout.addWidget(self.btn_add_mean)
        initial_means_layout.addLayout(im_header_layout)
        
        # Scroll area for initial means
        scroll_widget = QWidget()
        self.initial_means_scroll_layout = QVBoxLayout(scroll_widget)
        
        # Column headers
        headers_layout = QHBoxLayout()
        headers_layout.addWidget(QLabel("<b>Class Name</b>"), 2)
        headers_layout.addWidget(QLabel("<b>Blue</b>"), 1)
        headers_layout.addWidget(QLabel("<b>Green</b>"), 1)
        headers_layout.addWidget(QLabel("<b>Red</b>"), 1)
        noise_header = QLabel("<b>Noise</b>")
        noise_header.setToolTip("Mark as noise - component will be removed after training")
        headers_layout.addWidget(noise_header, 0)
        headers_layout.addWidget(QLabel(""), 0)  # For delete button
        self.initial_means_scroll_layout.addLayout(headers_layout)
        
        # Container for mean rows
        self.initial_means_rows = []
        
        scroll_area = QScrollArea()
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setMaximumHeight(200)
        initial_means_layout.addWidget(scroll_area)
        
        initial_means_group.setLayout(initial_means_layout)
        initial_means_main_layout.addWidget(initial_means_group)
        initial_means_widget.setLayout(initial_means_main_layout)
        
        # Add to vertical splitter
        v_splitter.addWidget(plot_widget)
        v_splitter.addWidget(initial_means_widget)
        v_splitter.setSizes([600, 200])  # More space to plots
        
        right_main_layout.addWidget(v_splitter)
        right_widget.setLayout(right_main_layout)
        
        # Add widgets to horizontal splitter
        h_splitter.addWidget(left_widget)
        h_splitter.addWidget(right_widget)
        h_splitter.setSizes([300, 1200])  # 1/5 left, 4/5 right
        
        main_layout.addWidget(h_splitter)
        
        self.setLayout(main_layout)
    
    def auto_populate_paths(self, material_folder):
        """Automatically populate paths based on material folder structure"""
        # Set images directory
        images_dir = os.path.join(material_folder, "images")
        self.image_dir_edit.setText(images_dir)
        
        # Set masks directory
        masks_dir = os.path.join(material_folder, "masks")
        self.mask_dir_edit.setText(masks_dir)
        
        # Set flatfield path
        flatfield_path = os.path.join(material_folder, "flatfield.png")
        if os.path.exists(flatfield_path):
            self.flatfield_edit.setText(flatfield_path)
        else:
            self.flatfield_edit.setText("Not found")
        
        # Set save directory
        save_dir = os.path.join(material_folder, "GMM")
        self.save_dir_edit.setText(save_dir)
        
        self.log_text.append(f"Auto-populated paths from: {material_folder}")
        self.log_text.append(f"Images: {images_dir}")
        self.log_text.append(f"Masks: {masks_dir}")
        self.log_text.append(f"Flatfield: {flatfield_path if os.path.exists(flatfield_path) else 'Not available'}")
        self.log_text.append(f"Save to: {save_dir}")
    
    def on_bounds_changed(self):
        """Update plot when cropping bounds change"""
        if self.contrast_data is not None:
            self.refresh_plot()
    
    def on_channels_changed(self):
        """Update plot when channel selection changes"""
        if self.contrast_data is not None:
            self.refresh_plot()
    
    def auto_crop_bounds(self):
        """Automatically set cropping bounds based on min/max RGB contrast values with padding"""
        if self.contrast_data is None:
            self.log_text.append("No contrast data loaded - cannot auto-crop")
            return
        
        # Get the padding percentage
        padding_percent = self.auto_crop_padding_spin.value() / 100.0
        
        # Calculate min/max for each channel (B, G, R)
        min_vals = np.min(self.contrast_data, axis=0)
        max_vals = np.max(self.contrast_data, axis=0)
        
        # Calculate the range for each channel
        ranges = max_vals - min_vals
        
        # Apply padding
        lower_bounds = min_vals - (ranges * padding_percent)
        upper_bounds = max_vals + (ranges * padding_percent)
        
        # Block signals to prevent multiple plot updates
        self.lower_b_spin.blockSignals(True)
        self.lower_g_spin.blockSignals(True)
        self.lower_r_spin.blockSignals(True)
        self.upper_b_spin.blockSignals(True)
        self.upper_g_spin.blockSignals(True)
        self.upper_r_spin.blockSignals(True)
        
        # Set the spinbox values
        self.lower_b_spin.setValue(round(lower_bounds[0], 2))
        self.lower_g_spin.setValue(round(lower_bounds[1], 2))
        self.lower_r_spin.setValue(round(lower_bounds[2], 2))
        self.upper_b_spin.setValue(round(upper_bounds[0], 2))
        self.upper_g_spin.setValue(round(upper_bounds[1], 2))
        self.upper_r_spin.setValue(round(upper_bounds[2], 2))
        
        # Unblock signals
        self.lower_b_spin.blockSignals(False)
        self.lower_g_spin.blockSignals(False)
        self.lower_r_spin.blockSignals(False)
        self.upper_b_spin.blockSignals(False)
        self.upper_g_spin.blockSignals(False)
        self.upper_r_spin.blockSignals(False)
        
        self.log_text.append(f"Auto-cropped bounds with {padding_percent*100:.0f}% padding:")
        self.log_text.append(f"  Lower: B={lower_bounds[0]:.2f}, G={lower_bounds[1]:.2f}, R={lower_bounds[2]:.2f}")
        self.log_text.append(f"  Upper: B={upper_bounds[0]:.2f}, G={upper_bounds[1]:.2f}, R={upper_bounds[2]:.2f}")
        
        # Update plot with new bounds
        self.refresh_plot()
    
    def on_canvas_press(self, event):
        """Handle mouse button press on canvas"""
        # Check for double-click to add new mean
        if event.dblclick and event.inaxes is not None and event.xdata is not None:
            self.add_mean_from_click(event)
            return
        
        # Check if any marker was clicked for dragging
        if event.inaxes is None or event.xdata is None:
            return
        
        # Find if we clicked on any marker
        for marker_info in self.draggable_markers:
            if marker_info['ax'] != event.inaxes:
                continue
            
            contains, _ = marker_info['scatter'].contains(event)
            if contains:
                # Start dragging this marker
                self.currently_dragging = marker_info.copy()
                self.currently_dragging['start_pos'] = (event.xdata, event.ydata)
                self.currently_dragging['orig_pos'] = marker_info['scatter'].get_offsets()[0].copy()
                break
    
    def on_canvas_motion(self, event):
        """Handle mouse motion on canvas"""
        if self.currently_dragging is None:
            return
        
        if event.xdata is None or event.ydata is None:
            return
        
        if event.inaxes != self.currently_dragging['ax']:
            return
        
        # Calculate new position
        dx = event.xdata - self.currently_dragging['start_pos'][0]
        dy = event.ydata - self.currently_dragging['start_pos'][1]
        
        new_x = self.currently_dragging['orig_pos'][0] + dx
        new_y = self.currently_dragging['orig_pos'][1] + dy
        
        # Update marker position
        self.currently_dragging['scatter'].set_offsets([[new_x, new_y]])
        self.canvas.draw_idle()
    
    def on_canvas_release(self, event):
        """Handle mouse button release on canvas"""
        if self.currently_dragging is None:
            return
        
        # Get final position
        final_pos = self.currently_dragging['scatter'].get_offsets()[0]
        
        # Update spinboxes with new position
        mean_idx = self.currently_dragging['mean_idx']
        channel_i = self.currently_dragging['channel_i']
        channel_j = self.currently_dragging['channel_j']
        
        if mean_idx < len(self.initial_means_rows):
            row = self.initial_means_rows[mean_idx]
            
            # Block signals to prevent recursive updates
            row['b'].blockSignals(True)
            row['g'].blockSignals(True)
            row['r'].blockSignals(True)
            
            # Update the appropriate channels
            if channel_i == 0:
                row['b'].setValue(float(final_pos[0]))
            elif channel_i == 1:
                row['g'].setValue(float(final_pos[0]))
            elif channel_i == 2:
                row['r'].setValue(float(final_pos[0]))
            
            if channel_j == 0:
                row['b'].setValue(float(final_pos[1]))
            elif channel_j == 1:
                row['g'].setValue(float(final_pos[1]))
            elif channel_j == 2:
                row['r'].setValue(float(final_pos[1]))
            
            # Unblock signals
            row['b'].blockSignals(False)
            row['g'].blockSignals(False)
            row['r'].blockSignals(False)
            
            # Update all other markers for this mean
            self.update_all_markers_for_mean(mean_idx)
        
        self.currently_dragging = None
    
    def update_all_markers_for_mean(self, mean_idx):
        """Update all subplot markers for a given mean index"""
        if mean_idx >= len(self.initial_means_rows):
            return
        
        row = self.initial_means_rows[mean_idx]
        mean = [row['b'].value(), row['g'].value(), row['r'].value()]
        
        # Update all markers for this mean index
        for marker_info in self.draggable_markers:
            if marker_info['mean_idx'] == mean_idx:
                i = marker_info['channel_i']
                j = marker_info['channel_j']
                marker_info['scatter'].set_offsets([[mean[i], mean[j]]])
        
        self.canvas.draw_idle()
    
    def add_mean_from_click(self, event):
        """Add a new initial mean from a click on the plot
        
        Args:
            event: Matplotlib mouse event
        """
        # Get which subplot was clicked and which channels it represents
        used_channels = self.used_channels_combo.currentText()
        used_channel_indices = ["BGR".index(channel) for channel in used_channels]
        import itertools
        channel_combinations = list(itertools.combinations(used_channel_indices, 2))
        
        # Find which subplot was clicked
        axes = self.figure.get_axes()
        subplot_idx = None
        for idx, ax in enumerate(axes):
            if ax == event.inaxes:
                subplot_idx = idx
                break
        
        if subplot_idx is None or subplot_idx >= len(channel_combinations):
            return
        
        # Get the channel indices for this subplot
        i, j = channel_combinations[subplot_idx]
        
        # Create a new initial mean with the clicked coordinates
        # We need to infer the third channel value (set to 0.0 for now)
        mean_bgr = [0.0, 0.0, 0.0]
        mean_bgr[i] = event.xdata
        mean_bgr[j] = event.ydata
        
        # Add a new row with these values
        self.add_initial_mean_row(mean_bgr[0], mean_bgr[1], mean_bgr[2])
    
    def refresh_plot(self):
        """Refresh the plot with current settings"""
        if self.contrast_data is not None:
            self.update_plot()
    
    def update_plot(self):
        """Update the plot with current contrast data and settings"""
        try:
            # Add scripts to path
            scripts_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts")
            if scripts_path not in sys.path:
                sys.path.append(scripts_path)
            
            from scripts.plotting_functions import create_heatmap, create_heatmap_plot
            import itertools
            import matplotlib.cm as cm
            
            # Clean up existing markers
            self.draggable_markers.clear()
            self.currently_dragging = None
            
            # Clear previous plot
            self.figure.clear()
            
            # Create heatmap plot manually on our figure
            axis_names = ["Blue Contrast", "Green Contrast", "Red Contrast"]
            used_channels = self.used_channels_combo.currentText()
            
            # Always use the cropping bounds
            upper_bounds = [
                self.upper_b_spin.value(),
                self.upper_g_spin.value(),
                self.upper_r_spin.value()
            ]
            lower_bounds = [
                self.lower_b_spin.value(),
                self.lower_g_spin.value(),
                self.lower_r_spin.value()
            ]
            
            # Extract the used channels and all possible combinations
            used_channel_indices = ["BGR".index(channel) for channel in used_channels]
            channel_combinations = list(itertools.combinations(used_channel_indices, 2))
            
            # Create subplots on our existing figure
            axes = []
            for idx in range(len(channel_combinations)):
                ax = self.figure.add_subplot(1, len(channel_combinations), idx + 1)
                axes.append(ax)
            
            # Plot each combination
            for idx, (i, j) in enumerate(channel_combinations):
                x = self.contrast_data[:, i]
                y = self.contrast_data[:, j]
                
                x_lower = lower_bounds[i]
                x_upper = upper_bounds[i]
                y_lower = lower_bounds[j]
                y_upper = upper_bounds[j]
                img, extent = create_heatmap(
                    x, y,
                    sigma=3,
                    bins=200,
                    extent=[x_lower, x_upper, y_lower, y_upper],
                )
                
                # Apply log scaling
                img = np.log(img + 1)
                
                axes[idx].imshow(
                    img, extent=extent, origin="lower", cmap=cm.plasma, aspect="auto"
                )
                
                axes[idx].set_xlabel(axis_names[i], fontsize=10)
                axes[idx].set_ylabel(axis_names[j], fontsize=10)
                axes[idx].grid(alpha=0.3)  # Reduced alpha for grid
                axes[idx].tick_params(labelsize=8)
                
                # Plot Gaussian ellipses if available
                if self.gmm_means is not None and self.gmm_covariances is not None:
                    from scripts.plotting_functions import confidence_ellipse
                    
                    for comp_idx in range(len(self.gmm_means)):
                        mean = self.gmm_means[comp_idx]
                        cov = self.gmm_covariances[comp_idx]
                        
                        # Extract 2D mean and covariance for this pair of channels
                        mean_2d = np.array([mean[i], mean[j]])
                        cov_2d = np.array([[cov[i, i], cov[i, j]], 
                                          [cov[j, i], cov[j, j]]])
                        
                        # Plot confidence ellipse (1 std deviation)
                        confidence_ellipse(
                            axes[idx],
                            mean_2d,
                            cov_2d,
                            n_std=1.0,
                            facecolor='none',
                            edgecolor='white'
                        )
                        
                        # Plot mean as a cross
                        axes[idx].scatter(mean[i], mean[j], 
                                        marker='x', s=100, c='white', 
                                        linewidths=2, zorder=10)
            
            # Plot initial means as scatter plots
            initial_means, _, noise_flags = self.get_initial_means_and_names()
            if initial_means is not None and len(initial_means) > 0:
                # Store marker artists for each subplot
                for idx, (i, j) in enumerate(channel_combinations):
                    for mean_idx, mean in enumerate(initial_means):
                        # Get color for this component (gray if marked as noise)
                        is_noise = noise_flags[mean_idx] if noise_flags else False
                        if is_noise:
                            face_color, edge_color = '#808080', '#404040'  # Gray for noise
                        else:
                            face_color, edge_color = MARKER_COLORS[mean_idx % len(MARKER_COLORS)]
                        
                        # Create scatter plot marker
                        scatter = axes[idx].scatter([mean[i]], [mean[j]], 
                                                   marker='*', s=200,
                                                   c=face_color, edgecolors=edge_color,
                                                   linewidths=1.5, zorder=15, alpha=0.9,
                                                   picker=10,  # Enable picking with 10 point tolerance
                                                   gid=f'mean_{mean_idx}_subplot_{idx}')  # Unique ID
                        self.draggable_markers.append({
                            'scatter': scatter,
                            'mean_idx': mean_idx,
                            'subplot_idx': idx,
                            'ax': axes[idx],
                            'channel_i': i,
                            'channel_j': j
                        })
            
            self.figure.suptitle("Contrast Distribution (Drag colored stars to adjust initial means)", fontsize=12)
            self.figure.tight_layout()
            
            # Refresh canvas
            self.canvas.draw()
            
        except Exception as e:
            self.log_text.append(f"Error updating plot: {e}")
            import traceback
            traceback.print_exc()
    
    def load_and_plot_data(self):
        """Load contrast data and plot heatmap"""
        # Get paths
        image_dir = self.image_dir_edit.text().strip()
        mask_dir = self.mask_dir_edit.text().strip()
        flatfield_path = self.flatfield_edit.text().strip()
        
        if not image_dir or not os.path.exists(image_dir):
            self.log_text.append("Warning: Images directory does not exist")
            return
            
        if not mask_dir or not os.path.exists(mask_dir):
            self.log_text.append("Warning: Masks directory does not exist")
            return
        
        # Check flatfield
        use_flatfield = None
        if flatfield_path and flatfield_path != "Not found" and os.path.exists(flatfield_path):
            use_flatfield = flatfield_path
        
        self.log_text.append("\n=== Loading Contrast Data ===")
        
        # Show progress dialog
        self.progress_dialog = QProgressDialog("Extracting contrast data...", None, 0, 0, self)
        self.progress_dialog.setWindowTitle("Loading")
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setMinimumDuration(0)
        self.progress_dialog.setCancelButton(None)
        self.progress_dialog.show()
        QApplication.processEvents()
        
        # Start loader thread
        self.loader_thread = GMMContrastLoaderThread(image_dir, mask_dir, use_flatfield)
        self.loader_thread.progress_update.connect(self.on_gmm_loading_progress)
        self.loader_thread.finished.connect(self.on_gmm_loading_finished)
        self.loader_thread.error.connect(self.on_gmm_loading_error)
        self.loader_thread.start()
    
    def on_gmm_loading_progress(self, message):
        """Update progress dialog with message"""
        if self.progress_dialog:
            self.progress_dialog.setLabelText(message)
            QApplication.processEvents()
        self.log_text.append(message)
    
    def on_gmm_loading_finished(self, contrast_data):
        """Handle completed contrast data loading"""
        if self.progress_dialog:
            self.progress_dialog.close()
            self.progress_dialog = None
        
        self.contrast_data = contrast_data
        self.log_text.append(f"Loaded {len(self.contrast_data)} contrast datapoints")
        
        # Auto-crop bounds based on loaded data
        self.auto_crop_bounds()
        
        # Update the plot
        self.update_plot()
        
        self.log_text.append("Heatmap plot generated successfully!")
    
    def on_gmm_loading_error(self, error_msg):
        """Handle loading error"""
        if self.progress_dialog:
            self.progress_dialog.close()
            self.progress_dialog = None
        
        self.log_text.append(f"Error loading data: {error_msg}")
        import traceback
        traceback.print_exc()
    
    def show_help(self):
        """Show help dialog for GMM training"""
        help_text = """
        <h3>GMM Training Dashboard</h3>
        <p>This tool trains a Gaussian Mixture Model (GMM) for material classification based on contrast values.</p>
        <p><b>Contrast data is automatically loaded and plotted when the dialog opens.</b></p>
        
        <h4>Data Configuration</h4>
        <ul>
            <li><b>Images Directory:</b> Automatically set to material/images</li>
            <li><b>Masks Directory:</b> Automatically set to material/masks</li>
            <li><b>Flatfield Image:</b> Automatically set to material/flatfield.png (if available)</li>
            <li><b>Save Directory:</b> Where the GMM parameters will be saved</li>
        </ul>
        
        <h4>Contrast Visualization</h4>
        <p>The heatmap on the right shows the distribution of contrast values extracted from your images and masks.
        This helps you visualize the clustering of different material thicknesses.</p>
        
        <h4>GMM Parameters</h4>
        <ul>
            <li><b>Number of Components:</b> Expected number of thickness clusters (e.g., 9 for graphene layers)</li>
            <li><b>Noise Components:</b> Extra Gaussian components to fit noise (increase if Gaussians are too large)</li>
            <li><b>Sample Size:</b> Number of datapoints to sample for training (reduce if training is too slow)</li>
            <li><b>Covariance Type:</b> Type of covariance matrix (full, tied, or diag)</li>
            <li><b>Used Channels:</b> Color channels to use for GMM (BGR, BG, GR, or BR)</li>
        </ul>
        
        <h4>Data Cropping</h4>
        <p>Define the contrast value range for visualization and training. The plot updates instantly as you adjust the bounds.</p>
        <ul>
            <li><b>Auto-Crop:</b> Automatically calculates optimal bounds based on min/max RGB values in your data</li>
            <li><b>Padding %:</b> Adds extra margin beyond the min/max values (default 5%)</li>
            <li><b>Lower/Upper Bounds:</b> Set the contrast range for each channel (B, G, R)</li>
            <li>Bounds are automatically set when data is loaded</li>
            <li>Adjust these values while viewing the heatmap to focus on specific contrast ranges</li>
            <li>Data outside these bounds will be excluded from training</li>
        </ul>
        
        <h4>Initial Means (Optional) - Interactive!</h4>
        <p>Configure initial Gaussian means to guide the fitting process and assign class names.</p>
        <ul>
            <li><b>Double-click on plot:</b> Add a new initial mean at the clicked position</li>
            <li><b>Drag colored stars:</b> Click and drag markers to adjust their position interactively</li>
            <li><b>Color-coded markers:</b> Each component has a unique color for easy identification</li>
            <li><b>Add Component button:</b> Add a new component at (0, 0, 0) manually</li>
            <li><b>Class Name:</b> Name for this component (used in GMM_parameters.json)</li>
            <li><b>BGR Values:</b> Adjust manually or by dragging markers on the plot</li>
            <li><b>Flexible:</b> Provide 0 to N initial means (where N = total components)</li>
            <li>You don't need to provide initial means for all components - only the ones you want to specify</li>
            <li>Remaining components (including noise if configured) are initialized randomly</li>
            <li>Markers update instantly when you change values or drag them</li>
            <li>Leave empty to use automatic k-means initialization for all components</li>
            <li><b>Delete button (×):</b> Remove an initial mean</li>
        </ul>
        
        <h4>Tips</h4>
        <ul>
            <li>Double-click on bright spots in the heatmap to place initial means accurately</li>
            <li>Drag markers to fine-tune positions visually</li>
            <li>Each marker has a unique color to distinguish different components</li>
            <li>Provide initial means only for components you can clearly identify</li>
            <li>If you see 10 clusters, set components to 10 and provide up to 10 initial means</li>
            <li>Noise components are just regular Gaussians - they're interpreted as noise after training</li>
            <li>If Gaussians are too large, increase noise components to catch outliers</li>
            <li>If training takes too long, reduce sample size (30,000 is recommended)</li>
            <li>At least 5 flakes per thickness with ~1000px each is recommended</li>
            <li>Use data cropping to focus on specific contrast ranges</li>
        </ul>
        """
        show_help_dialog(self, "GMM Training Help", help_text)
    
    def start_training(self):
        """Start GMM training with current parameters"""
        # Validate inputs
        image_dir = self.image_dir_edit.text().strip()
        mask_dir = self.mask_dir_edit.text().strip()
        save_dir = self.save_dir_edit.text().strip()
        
        if not image_dir or not os.path.exists(image_dir):
            QMessageBox.warning(self, "Error", "Images directory does not exist")
            return
            
        if not mask_dir or not os.path.exists(mask_dir):
            QMessageBox.warning(self, "Error", "Masks directory does not exist")
            return
            
        if not save_dir:
            QMessageBox.warning(self, "Error", "Please specify a save directory")
            return
        
        # Check flatfield
        flatfield_path = self.flatfield_edit.text().strip()
        if flatfield_path and flatfield_path != "Not found" and os.path.exists(flatfield_path):
            use_flatfield = flatfield_path
        else:
            use_flatfield = None
            self.log_text.append("Note: No flatfield correction will be applied")
        
        # Collect parameters
        params = {
            'num_components': self.num_components_spin.value(),
            'num_noise_components': self.num_noise_spin.value(),
            'sample_size': self.sample_size_spin.value(),
            'cov_type': self.cov_type_combo.currentText(),
            'used_channels': self.used_channels_combo.currentText(),
            'lower_bounds': [
                self.lower_b_spin.value(),
                self.lower_g_spin.value(),
                self.lower_r_spin.value()
            ],
            'upper_bounds': [
                self.upper_b_spin.value(),
                self.upper_g_spin.value(),
                self.upper_r_spin.value()
            ]
        }
        
        # Clear log and reset progress
        self.log_text.clear()
        self.progress_bar.setValue(0)
        
        # Get initial means, class names, and noise flags if configured
        initial_means, class_names, noise_flags = self.get_initial_means_and_names()
        
        # Validate initial means count doesn't exceed total components
        if initial_means is not None:
            total_components = self.num_components_spin.value()
            if len(initial_means) > total_components:
                QMessageBox.warning(
                    self, "Error", 
                    f"Number of initial means ({len(initial_means)}) cannot exceed "
                    f"total number of components ({total_components}).\n\n"
                    f"You can provide 0 to {total_components} initial means.\n"
                    f"Remaining components will be initialized randomly."
                )
                return
            
            self.log_text.append(f"Using {len(initial_means)} user-provided initial means")
            if len(initial_means) < total_components:
                num_random = total_components - len(initial_means)
                self.log_text.append(f"Remaining {num_random} component(s) will be initialized randomly")
            
            # Log noise components
            if noise_flags:
                num_noise = sum(noise_flags)
                if num_noise > 0:
                    noise_names = [class_names[i] for i, is_noise in enumerate(noise_flags) if is_noise]
                    self.log_text.append(f"Components marked as noise ({num_noise}): {', '.join(noise_names)}")
                    self.log_text.append("These will be removed from the final model after training.")
        
        # Disable start button, enable stop button
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        
        # Create and start training thread
        self.training_thread = GMMTrainingThread(
            image_dir, mask_dir, save_dir, params, use_flatfield, initial_means, class_names, noise_flags
        )
        self.training_thread.progress_update.connect(self.update_progress)
        self.training_thread.log_message.connect(self.append_log)
        self.training_thread.training_complete.connect(self.training_finished)
        self.training_thread.plot_gaussians.connect(self.plot_trained_gaussians)
        self.training_thread.start()
    
    def stop_training(self):
        """Stop the current training"""
        if self.training_thread and self.training_thread.isRunning():
            self.training_thread.stop()
            self.log_text.append("Stopping training...")
            self.btn_stop.setEnabled(False)
    
    def update_progress(self, value):
        """Update progress bar"""
        self.progress_bar.setValue(value)
    
    def append_log(self, message):
        """Append message to log"""
        self.log_text.append(message)
    
    def training_finished(self, success):
        """Handle training completion"""
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        
        if success:
            QMessageBox.information(self, "Success", "GMM training completed successfully!")
        else:
            QMessageBox.warning(self, "Error", "GMM training failed. Check the log for details.")
    
    def plot_trained_gaussians(self, means, covariances):
        """Store GMM parameters and update plot with ellipses"""
        self.gmm_means = means
        self.gmm_covariances = covariances
        self.log_text.append("Plotting Gaussian ellipses...")
        # Update the plot to show ellipses
        self.update_plot()
    
    def add_initial_mean_row(self, b_val=0.0, g_val=0.0, r_val=0.0):
        """Add a row for configuring an initial mean
        
        Args:
            b_val: Initial blue value
            g_val: Initial green value
            r_val: Initial red value
        """
        row_layout = QHBoxLayout()
        
        # Class name
        name_edit = QLineEdit()
        name_edit.setPlaceholderText(f"Class {len(self.initial_means_rows) + 1}")
        row_layout.addWidget(name_edit, 2)
        
        # Blue value
        b_spin = QDoubleSpinBox()
        b_spin.setRange(-5.0, 5.0)
        b_spin.setValue(b_val)
        b_spin.setSingleStep(0.01)
        b_spin.setDecimals(3)
        b_spin.valueChanged.connect(self.on_initial_means_changed)
        row_layout.addWidget(b_spin, 1)
        
        # Green value
        g_spin = QDoubleSpinBox()
        g_spin.setRange(-5.0, 5.0)
        g_spin.setValue(g_val)
        g_spin.setSingleStep(0.01)
        g_spin.setDecimals(3)
        g_spin.valueChanged.connect(self.on_initial_means_changed)
        row_layout.addWidget(g_spin, 1)
        
        # Red value
        r_spin = QDoubleSpinBox()
        r_spin.setRange(-5.0, 5.0)
        r_spin.setValue(r_val)
        r_spin.setSingleStep(0.01)
        r_spin.setDecimals(3)
        r_spin.valueChanged.connect(self.on_initial_means_changed)
        row_layout.addWidget(r_spin, 1)
        
        # Noise checkbox - marks component for removal after training
        noise_check = QCheckBox()
        noise_check.setToolTip("Mark as noise - component will be removed after training")
        noise_check.stateChanged.connect(self.on_initial_means_changed)
        row_layout.addWidget(noise_check, 0)
        
        # Delete button
        btn_delete = QPushButton("×")
        btn_delete.setMaximumWidth(30)
        btn_delete.setStyleSheet("color: red; font-weight: bold; font-size: 16px;")
        btn_delete.clicked.connect(lambda: self.remove_initial_mean_row(row_layout))
        row_layout.addWidget(btn_delete, 0)
        
        # Store the row
        self.initial_means_rows.append({
            'layout': row_layout,
            'name': name_edit,
            'b': b_spin,
            'g': g_spin,
            'r': r_spin,
            'noise': noise_check
        })
        
        # Add to scroll layout
        self.initial_means_scroll_layout.addLayout(row_layout)
        
        # Update plot to show the new marker
        if self.contrast_data is not None:
            self.update_plot()
    
    def remove_initial_mean_row(self, row_layout):
        """Remove an initial mean row"""
        # Find and remove the row
        for i, row_data in enumerate(self.initial_means_rows):
            if row_data['layout'] == row_layout:
                # Remove widgets from layout
                while row_layout.count():
                    item = row_layout.takeAt(0)
                    if item.widget():
                        item.widget().deleteLater()
                
                # Remove layout
                self.initial_means_scroll_layout.removeItem(row_layout)
                
                # Remove from list
                self.initial_means_rows.pop(i)
                break
        
        # Update plot to remove the marker
        if self.contrast_data is not None:
            self.update_plot()
    
    def on_initial_means_changed(self):
        """Update plot when initial means values change"""
        if self.contrast_data is not None:
            self.update_plot()
    
    def get_initial_means_and_names(self):
        """Get configured initial means, class names, and noise flags
        
        Returns:
            tuple: (means_array, names_list, noise_flags_list)
        """
        if not self.initial_means_rows:
            return None, None, None
        
        means = []
        names = []
        noise_flags = []
        
        for row_data in self.initial_means_rows:
            # Get BGR values
            b = row_data['b'].value()
            g = row_data['g'].value()
            r = row_data['r'].value()
            means.append([b, g, r])
            
            # Get class name
            name = row_data['name'].text().strip()
            if not name:
                name = row_data['name'].placeholderText()
            names.append(name)
            
            # Get noise flag
            noise_flags.append(row_data['noise'].isChecked())
        
        return np.array(means) if means else None, names if names else None, noise_flags if noise_flags else None


