import os
import sys
import numpy as np
import torch
import torch.nn as nn
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QLineEdit, QSpinBox, QDoubleSpinBox, 
                             QGroupBox, QGridLayout, QFileDialog, QTextEdit, 
                             QProgressBar, QMessageBox, QCheckBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from gui.help_dialog import show_help_dialog


class AMMTrainingThread(QThread):
    """Separate thread for AMM training based on train_AMM_interactive.ipynb"""
    progress_update = pyqtSignal(int)
    log_message = pyqtSignal(str)
    training_complete = pyqtSignal(bool)
    plot_distribution = pyqtSignal(object, object, object, object)  # model, dataloader, loc, cov
    
    def __init__(self, train_image_dir, train_annotation_path, save_dir, params):
        super().__init__()
        self.train_image_dir = train_image_dir
        self.train_annotation_path = train_annotation_path
        self.save_dir = save_dir
        self.params = params
        self.should_stop = False
        
    def run(self):
        """Run AMM training exactly like the interactive notebook"""
        try:
            self.log_message.emit("Starting AMM training...")
            
            # Add maskterial to path
            maskterial_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "maskterial")
            if maskterial_path not in sys.path:
                sys.path.append(maskterial_path)
                
            from maskterial.modeling.common.fcresnet import FCResNet
            from maskterial.utils.data_loader import ContrastDataloader
            
            self.progress_update.emit(10)
            self.log_message.emit("Loading data with ContrastDataloader...")
            
            # Set random seeds for reproducibility
            np.random.seed(42)
            torch.manual_seed(42)
            
            # Create dataloader exactly like notebook
            dataloader = ContrastDataloader(
                train_image_dir=self.train_image_dir,
                train_annotation_path=self.train_annotation_path,
                test_image_dir=None,
                test_annotation_path=None,
                max_samples_per_class=self.params['max_samples_per_class'],
                loaded_test_samples=self.params['loaded_test_samples'],
                uniform_class_sampling=self.params['uniform_class_sampling'],
                use_normalization=self.params['use_normalization'],
                use_DBSCAN=self.params['use_dbscan'],
                DBSCAN_eps=self.params['dbscan_eps'],
                use_Nearest_Neighbors=self.params['use_nearest_neighbors'],
                neighbors=self.params['neighbors'],
                verbose=True
            )
            
            num_classes = dataloader.num_classes
            self.progress_update.emit(30)
            self.log_message.emit(f"Loaded data: {num_classes} classes")
            
            if self.should_stop:
                return
            
            # Create model exactly like notebook
            model = FCResNet(
                input_dim=self.params['input_dim'],
                embedding_dim=self.params['embedding_dim'],
                depth=self.params['depth'],
                num_classes=num_classes,
                spec_coeff=self.params['spec_coeff'],
                n_power_iterations=self.params['n_power_iterations'],
                spectral_normalization=self.params['spectral_normalization'],
                dropout_rate=self.params['dropout_rate']
            )
            
            # Set up training exactly like notebook
            optimizer = torch.optim.Adam(model.parameters(), lr=self.params['lr'])
            loss_function = nn.CrossEntropyLoss()
            
            self.progress_update.emit(40)
            self.log_message.emit("Starting training loop...")
            
            # Initialize tracking variables
            final_loss = 0.0
            final_accuracy = 0.0
            
            # Training loop exactly like notebook
            for iteration in range(self.params['num_iter']):
                if self.should_stop:
                    break
                    
                model.train()
                
                # Get batch exactly like notebook
                X_train, y_train = dataloader.get_batch(batch_size=self.params['batch_size'])
                X_train = torch.tensor(X_train, dtype=torch.float32)
                y_train = torch.tensor(y_train, dtype=torch.int64)
                
                # Forward pass
                logits = model(X_train)
                loss = loss_function(logits, y_train)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                # Store final loss and calculate accuracy for the last iteration
                final_loss = loss.item()
                if iteration == self.params['num_iter'] - 1 or iteration % self.params['test_interval'] == 0:
                    # Calculate accuracy
                    with torch.no_grad():
                        predictions = torch.argmax(logits, dim=1)
                        correct = (predictions == y_train).float()
                        final_accuracy = correct.mean().item() * 100
                
                # Progress and logging
                if iteration % self.params['test_interval'] == 0:
                    train_loss = loss.item()
                    progress = 40 + int((iteration / self.params['num_iter']) * 50)
                    self.progress_update.emit(progress)
                    self.log_message.emit(f"Iteration: {iteration:5d} Train Loss: {train_loss:10.5f} Accuracy: {final_accuracy:.1f}%")
            
            self.progress_update.emit(90)
            self.log_message.emit("Calculating class embeddings...")
            
            # Calculate class embeddings exactly like notebook
            loc, cov = self.calculate_class_embeddings(model, dataloader)
            
            self.progress_update.emit(95)
            self.log_message.emit("Saving model...")
            
            # Save model and parameters
            os.makedirs(self.save_dir, exist_ok=True)
            
            # Save model state
            torch.save(model.state_dict(), os.path.join(self.save_dir, "model.pth"))
            
            # Save AMM parameters
            np.save(os.path.join(self.save_dir, "loc.npy"), loc.cpu().numpy())
            np.save(os.path.join(self.save_dir, "cov.npy"), cov.cpu().numpy())
            
            # Save config
            import json
            config = {
                'model_config': {
                    'input_dim': self.params['input_dim'],
                    'num_classes': num_classes,
                    'embedding_dim': self.params['embedding_dim'],
                    'depth': self.params['depth'],
                },
                'num_classes': num_classes
            }
            with open(os.path.join(self.save_dir, "config.json"), 'w') as f:
                json.dump(config, f, indent=2)
            
            # Save comprehensive metadata in the expected format
            self.log_message.emit("Saving metadata...")
            from datetime import datetime
            
            # Calculate train mean and std from dataloader if available
            train_mean = None
            train_std = None
            if hasattr(dataloader, 'X_train'):
                train_mean = np.mean(dataloader.X_train, axis=0).tolist()
                train_std = np.std(dataloader.X_train, axis=0).tolist()
            
            metadata = {
                "train_config": {
                    "train_params": {
                        "num_iterations": self.params['num_iter'],
                        "learning_rate": self.params['lr'],
                        "loss_function": "CrossEntropyLoss",
                        "test_interval": self.params['test_interval'],
                        "batch_size": self.params['batch_size']
                    },
                    "data_params": {
                        "max_samples_per_class": self.params['max_samples_per_class'],
                        "loaded_test_samples": self.params['loaded_test_samples'],
                        "uniform_class_sampling": self.params['uniform_class_sampling'],
                        "use_normalization": self.params['use_normalization'],
                        "use_DBSCAN": self.params['use_dbscan'],
                        "DBSCAN_eps": self.params['dbscan_eps'],
                        "use_Nearest_Neighbors": self.params['use_nearest_neighbors'],
                        "neighbors": self.params['neighbors']
                    },
                    "model_arch": {
                        "num_classes": num_classes,
                        "input_dim": self.params['input_dim'],
                        "embedding_dim": self.params['embedding_dim'],
                        "depth": self.params['depth'],
                        "spec_coeff": self.params['spec_coeff'],
                        "n_power_iterations": self.params['n_power_iterations'],
                        "spectral_normalization": self.params['spectral_normalization'],
                        "dropout_rate": self.params['dropout_rate']
                    }
                },
                "test_losses": {
                    "final": float('inf'),  # No test set evaluation in current implementation
                    "best": float('inf')
                },
                "train_mean": train_mean,
                "train_std": train_std,
                "train_image_dir": self.train_image_dir,
                "train_annotation_path": self.train_annotation_path,
                "test_image_dir": None,
                "test_annotation_path": None
            }
            
            with open(os.path.join(self.save_dir, "meta_data.json"), 'w') as f:
                json.dump(metadata, f, indent=4)
            
            # Show distribution plot if enabled
            if self.params.get('show_distribution_plot', False):
                self.progress_update.emit(98)
                self.log_message.emit("Generating distribution plot...")
                # Emit signal to plot in main thread instead of plotting directly
                self.plot_distribution.emit(model, dataloader, loc, cov)
            
            self.progress_update.emit(100)
            self.log_message.emit(f"Training completed! Model, metadata, and parameters saved to: {self.save_dir}")
            self.training_complete.emit(True)
            
        except Exception as e:
            self.log_message.emit(f"Error: {e}")
            import traceback
            traceback.print_exc()
            self.training_complete.emit(False)
    
    def calculate_class_embeddings(self, model, dataloader):
        """Calculate class embeddings exactly like notebook"""
        model.eval()
        with torch.no_grad():
            X_torch = torch.tensor(dataloader.X_train).float()
            input_embeddings = model.get_embedding(X_torch)
            
            # Calculate mean and covariance for each class
            loc = torch.stack([
                torch.mean(input_embeddings[dataloader.y_train == class_id], dim=0)
                for class_id in range(dataloader.num_classes)
            ])
            
            cov = torch.stack([
                torch.cov(input_embeddings[dataloader.y_train == class_id].T)
                for class_id in range(dataloader.num_classes)
            ])
            
            return loc, cov
    
    # Remove plot_distribution method from thread class - moved to main dialog


class AMMTrainingDialog(QDialog):
    def __init__(self, parent=None, project_folder=None):
        super().__init__(parent)
        self.training_thread = None
        self.project_folder = project_folder
        self.init_ui()
        
        # Pre-populate paths if project folder is provided
        if self.project_folder:
            images_dir = os.path.join(self.project_folder, "images")
            semantic_masks_dir = os.path.join(self.project_folder, "semantic_masks")
            amm_save_dir = os.path.join(self.project_folder, "AMM")
            
            if os.path.exists(images_dir):
                self.train_image_edit.setText(images_dir)
            if os.path.exists(semantic_masks_dir):
                self.train_masks_edit.setText(semantic_masks_dir)
            # Set save directory to dedicated AMM folder
            self.save_dir_edit.setText(amm_save_dir)
        
    def init_ui(self):
        """Initialize the AMM training UI based on notebook parameters"""
        self.setWindowTitle("AMM Training - Interactive")
        self.resize(800, 700)
        
        layout = QVBoxLayout()
        
        # Header with title and help button
        header_layout = QHBoxLayout()
        
        # Title
        title_label = QLabel("🧠 AMM Training - Gaussian Mixture Model")
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
        
        layout.addLayout(header_layout)
        
        # Data Configuration
        data_group = QGroupBox("Data Configuration")
        data_layout = QGridLayout()
        
        data_layout.addWidget(QLabel("Train Images Directory:"), 0, 0)
        self.train_image_edit = QLineEdit()
        self.train_image_edit.setPlaceholderText("Path to training images")
        data_layout.addWidget(self.train_image_edit, 0, 1)
        self.btn_browse_images = QPushButton("Browse")
        self.btn_browse_images.clicked.connect(self.browse_train_images)
        data_layout.addWidget(self.btn_browse_images, 0, 2)
        
        data_layout.addWidget(QLabel("Semantic Masks Directory:"), 1, 0)
        self.train_masks_edit = QLineEdit()
        self.train_masks_edit.setPlaceholderText("Path to semantic masks")
        data_layout.addWidget(self.train_masks_edit, 1, 1)
        self.btn_browse_masks = QPushButton("Browse")
        self.btn_browse_masks.clicked.connect(self.browse_train_masks)
        data_layout.addWidget(self.btn_browse_masks, 1, 2)
        
        data_layout.addWidget(QLabel("Save Directory:"), 2, 0)
        self.save_dir_edit = QLineEdit()
        self.save_dir_edit.setPlaceholderText("Where to save trained model")
        data_layout.addWidget(self.save_dir_edit, 2, 1)
        self.btn_browse_save = QPushButton("Browse")
        self.btn_browse_save.clicked.connect(self.browse_save_dir)
        data_layout.addWidget(self.btn_browse_save, 2, 2)
        
        data_group.setLayout(data_layout)
        layout.addWidget(data_group)
        
        # Training Parameters (from notebook)
        train_group = QGroupBox("Training Parameters")
        train_layout = QGridLayout()
        
        train_layout.addWidget(QLabel("Iterations:"), 0, 0)
        self.num_iter_spin = QSpinBox()
        self.num_iter_spin.setRange(100, 20000)
        self.num_iter_spin.setValue(5000)
        train_layout.addWidget(self.num_iter_spin, 0, 1)
        
        train_layout.addWidget(QLabel("Learning Rate:"), 0, 2)
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.001, 1.0)
        self.lr_spin.setValue(0.01)
        self.lr_spin.setDecimals(4)
        train_layout.addWidget(self.lr_spin, 0, 3)
        
        train_layout.addWidget(QLabel("Batch Size:"), 1, 0)
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1000, 50000)
        self.batch_size_spin.setValue(10000)
        train_layout.addWidget(self.batch_size_spin, 1, 1)
        
        train_layout.addWidget(QLabel("Test Interval:"), 1, 2)
        self.test_interval_spin = QSpinBox()
        self.test_interval_spin.setRange(1, 1000)
        self.test_interval_spin.setValue(10)
        train_layout.addWidget(self.test_interval_spin, 1, 3)
        
        train_group.setLayout(train_layout)
        layout.addWidget(train_group)
        
        # Data Processing Parameters (from notebook)
        data_proc_group = QGroupBox("Data Processing Parameters")
        data_proc_layout = QGridLayout()
        
        data_proc_layout.addWidget(QLabel("Max Samples/Class:"), 0, 0)
        self.max_samples_spin = QSpinBox()
        self.max_samples_spin.setRange(1000, 100000)
        self.max_samples_spin.setValue(30000)
        data_proc_layout.addWidget(self.max_samples_spin, 0, 1)
        
        data_proc_layout.addWidget(QLabel("DBSCAN Eps:"), 0, 2)
        self.dbscan_eps_spin = QDoubleSpinBox()
        self.dbscan_eps_spin.setRange(0.01, 1.0)
        self.dbscan_eps_spin.setValue(0.15)
        self.dbscan_eps_spin.setDecimals(3)
        data_proc_layout.addWidget(self.dbscan_eps_spin, 0, 3)
        
        data_proc_layout.addWidget(QLabel("Neighbors:"), 1, 0)
        self.neighbors_spin = QSpinBox()
        self.neighbors_spin.setRange(5, 100)
        self.neighbors_spin.setValue(25)
        data_proc_layout.addWidget(self.neighbors_spin, 1, 1)
        
        # Checkboxes for boolean parameters
        self.uniform_sampling_cb = QCheckBox("Uniform Class Sampling")
        self.uniform_sampling_cb.setChecked(True)
        data_proc_layout.addWidget(self.uniform_sampling_cb, 2, 0)
        
        self.use_normalization_cb = QCheckBox("Use Normalization")
        self.use_normalization_cb.setChecked(True)
        data_proc_layout.addWidget(self.use_normalization_cb, 2, 1)
        
        self.use_dbscan_cb = QCheckBox("Use DBSCAN")
        self.use_dbscan_cb.setChecked(True)
        data_proc_layout.addWidget(self.use_dbscan_cb, 2, 2)
        
        self.use_neighbors_cb = QCheckBox("Use Nearest Neighbors")
        self.use_neighbors_cb.setChecked(True)
        data_proc_layout.addWidget(self.use_neighbors_cb, 2, 3)
        
        data_proc_group.setLayout(data_proc_layout)
        layout.addWidget(data_proc_group)
        
        # Model Architecture (from notebook)
        model_group = QGroupBox("Model Architecture")
        model_layout = QGridLayout()
        
        model_layout.addWidget(QLabel("Input Dim:"), 0, 0)
        self.input_dim_spin = QSpinBox()
        self.input_dim_spin.setRange(1, 10)
        self.input_dim_spin.setValue(3)
        model_layout.addWidget(self.input_dim_spin, 0, 1)
        
        model_layout.addWidget(QLabel("Embedding Dim:"), 0, 2)
        self.embedding_dim_spin = QSpinBox()
        self.embedding_dim_spin.setRange(8, 128)
        self.embedding_dim_spin.setValue(16)
        model_layout.addWidget(self.embedding_dim_spin, 0, 3)
        
        model_layout.addWidget(QLabel("Depth:"), 1, 0)
        self.depth_spin = QSpinBox()
        self.depth_spin.setRange(1, 10)
        self.depth_spin.setValue(4)
        model_layout.addWidget(self.depth_spin, 1, 1)
        
        model_layout.addWidget(QLabel("Spec Coeff:"), 1, 2)
        self.spec_coeff_spin = QDoubleSpinBox()
        self.spec_coeff_spin.setRange(0.1, 2.0)
        self.spec_coeff_spin.setValue(0.5)
        self.spec_coeff_spin.setDecimals(2)
        model_layout.addWidget(self.spec_coeff_spin, 1, 3)
        
        model_layout.addWidget(QLabel("N Power Iterations:"), 2, 0)
        self.n_power_spin = QSpinBox()
        self.n_power_spin.setRange(1, 20)
        self.n_power_spin.setValue(5)
        model_layout.addWidget(self.n_power_spin, 2, 1)
        
        model_layout.addWidget(QLabel("Dropout Rate:"), 2, 2)
        self.dropout_spin = QDoubleSpinBox()
        self.dropout_spin.setRange(0.0, 0.9)
        self.dropout_spin.setValue(0.1)
        self.dropout_spin.setDecimals(2)
        model_layout.addWidget(self.dropout_spin, 2, 3)
        
        self.spectral_norm_cb = QCheckBox("Spectral Normalization")
        self.spectral_norm_cb.setChecked(True)
        model_layout.addWidget(self.spectral_norm_cb, 3, 0)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # Visualization Options
        viz_group = QGroupBox("Visualization Options")
        viz_layout = QHBoxLayout()
        
        self.show_distribution_cb = QCheckBox("Show Distribution Plot After Training")
        self.show_distribution_cb.setChecked(True)
        self.show_distribution_cb.setToolTip("Display 3D contrast distribution plot when training completes (may take extra time)")
        viz_layout.addWidget(self.show_distribution_cb)
        
        viz_group.setLayout(viz_layout)
        layout.addWidget(viz_group)
        
        # Progress and controls
        progress_layout = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        progress_layout.addWidget(self.progress_bar)
        layout.addLayout(progress_layout)
        
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
        
        layout.addLayout(buttons_layout)
        
        # Training log
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(200)
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)
        
        self.setLayout(layout)
        
    def browse_train_images(self):
        """Browse for training images directory"""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Training Images Directory")
        if dir_path:
            self.train_image_edit.setText(dir_path)
            
    def browse_train_masks(self):
        """Browse for training masks directory"""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Semantic Masks Directory")
        if dir_path:
            self.train_masks_edit.setText(dir_path)
            
    def browse_save_dir(self):
        """Browse for save directory"""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Save Directory")
        if dir_path:
            self.save_dir_edit.setText(dir_path)
        
    def start_training(self):
        """Start AMM training with current parameters"""
        # Validate inputs
        train_image_dir = self.train_image_edit.text().strip()
        train_masks_dir = self.train_masks_edit.text().strip()
        save_dir = self.save_dir_edit.text().strip()
        
        if not train_image_dir or not os.path.exists(train_image_dir):
            QMessageBox.warning(self, "Error", "Please select a valid training images directory")
            return
            
        if not train_masks_dir or not os.path.exists(train_masks_dir):
            QMessageBox.warning(self, "Error", "Please select a valid semantic masks directory")
            return
            
        if not save_dir:
            QMessageBox.warning(self, "Error", "Please specify a save directory")
            return
        
        # Collect all parameters exactly like notebook
        params = {
            # Training params
            'num_iter': self.num_iter_spin.value(),
            'lr': self.lr_spin.value(),
            'batch_size': self.batch_size_spin.value(),
            'test_interval': self.test_interval_spin.value(),
            
            # Data params
            'max_samples_per_class': self.max_samples_spin.value(),
            'loaded_test_samples': 50000,  # Fixed from notebook
            'uniform_class_sampling': self.uniform_sampling_cb.isChecked(),
            'use_normalization': self.use_normalization_cb.isChecked(),
            'use_dbscan': self.use_dbscan_cb.isChecked(),
            'dbscan_eps': self.dbscan_eps_spin.value(),
            'use_nearest_neighbors': self.use_neighbors_cb.isChecked(),
            'neighbors': self.neighbors_spin.value(),
            
            # Model architecture
            'input_dim': self.input_dim_spin.value(),
            'embedding_dim': self.embedding_dim_spin.value(),
            'depth': self.depth_spin.value(),
            'spec_coeff': self.spec_coeff_spin.value(),
            'n_power_iterations': self.n_power_spin.value(),
            'spectral_normalization': self.spectral_norm_cb.isChecked(),
            'dropout_rate': self.dropout_spin.value(),
            
            # Visualization options
            'show_distribution_plot': self.show_distribution_cb.isChecked()
        }
        
        # Start training thread
        self.training_thread = AMMTrainingThread(
            train_image_dir, train_masks_dir, save_dir, params
        )
        self.training_thread.progress_update.connect(self.update_progress)
        self.training_thread.log_message.connect(self.log_message)
        self.training_thread.training_complete.connect(self.on_training_complete)
        self.training_thread.plot_distribution.connect(self.plot_distribution)
        
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress_bar.setValue(0)
        self.log_text.clear()
        
        self.training_thread.start()
        
    def stop_training(self):
        """Stop the training process"""
        if self.training_thread and self.training_thread.isRunning():
            self.training_thread.should_stop = True
            self.training_thread.wait()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        
    def update_progress(self, value):
        """Update progress bar"""
        self.progress_bar.setValue(value)
        
    def log_message(self, message):
        """Add message to log"""
        self.log_text.append(message)
        self.log_text.ensureCursorVisible()
        
    def on_training_complete(self, success):
        """Handle training completion"""
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        
        if success:
            QMessageBox.information(self, "Success", "AMM training completed successfully!")
        else:
            QMessageBox.warning(self, "Error", "AMM training failed. Check the log for details.")
    
    def plot_distribution(self, model, dataloader, loc, cov, standard_deviation_levels=[1, 2, 3, 4, 5, 6]):
        """Plot the approximated contrast distribution with training data - runs in main thread"""
        import matplotlib.pyplot as plt
        
        try:
            self.log_message("Computing 3D distribution grid...")
            
            AXIS_NAMES = [
                "Normalized Blue Contrast",
                "Normalized Green Contrast", 
                "Normalized Red Contrast",
            ]
            
            # Get training data
            X = dataloader.X_train
            y = dataloader.y_train
            
            # Unique categories
            class_ids = np.unique(y)
            plot_colors = plt.cm.plasma(np.linspace(0, 1, len(class_ids)))
            
            # Normalize the covariance matrix to avoid numerical issues
            reg_cov = cov + 1e-5 * torch.eye(cov.shape[-1])
            
            # Create 3D grid for visualization
            X_sidelength = torch.arange(-3, 3, 0.05)
            X_cube = torch.stack(
                torch.meshgrid(X_sidelength, X_sidelength, X_sidelength, indexing="ij"),
                dim=-1,
            )
            X_cube_shape = X_cube.shape
            X_cube_flat = X_cube.reshape(-1, 3).float()
            
            # Compute embeddings for all possible contrast values
            model.eval()
            with torch.no_grad():
                X_cube_embeddings = model.get_embedding(X_cube_flat)
            
            # Compute Mahalanobis distances manually
            X_cube_mh_distances = []
            for class_idx in range(len(class_ids)):
                diff = X_cube_embeddings - loc[class_idx].unsqueeze(0)
                inv_cov = torch.inverse(reg_cov[class_idx])
                mh_dist = torch.sum(diff @ inv_cov * diff, dim=1)
                X_cube_mh_distances.append(mh_dist)
            
            X_cube_mh_distances = torch.stack(X_cube_mh_distances, dim=1)
            
            # Get minimum distances and reshape
            X_cube_label_distances, X_cube_labels = torch.min(X_cube_mh_distances, dim=-1)
            X_cube_label_distances = X_cube_label_distances.cpu().numpy()
            X_cube_label_distances = X_cube_label_distances.reshape(X_cube_shape[:-1])
            
            self.log_message("Creating distribution plots...")
            
            # Create the plot - now safe to run in main thread
            fig, axs = plt.subplots(1, 3, figsize=(15, 5), dpi=150)
            plt.rcParams.update({"font.size": 12})
            plt.rcParams.update({"axes.titlesize": 12})
            
            for i, ax in enumerate(axs):
                # Scatter plot of training data
                for class_id, color in zip(class_ids, plot_colors):
                    mask = y == class_id
                    ax.scatter(
                        X[mask, i],
                        X[mask, (i + 1) % 3],
                        color=color,
                        label=f"Class {class_id}",
                        s=15,
                        alpha=0.1,
                        ec=None,
                    )
                
                # Project 3D distance onto 2D plane
                if i == 2:
                    distance_min = X_cube_label_distances.min(axis=((i + 2) % 3))
                else:
                    distance_min = X_cube_label_distances.min(axis=((i + 2) % 3)).T
                
                # Draw contour plot
                ax.contour(
                    X_sidelength,
                    X_sidelength,
                    distance_min,
                    alpha=0.5,
                    levels=standard_deviation_levels,
                    cmap="viridis",
                )
                
                ax.set_xlabel(AXIS_NAMES[i])
                ax.set_ylabel(AXIS_NAMES[(i + 1) % 3])
                ax.set_xlim(-3, 3)
                ax.set_ylim(-3, 3)
                
                # Make legend look nice
                leg = ax.legend()
                for lh in leg.legend_handles:
                    lh.set_alpha(1)
                    lh.set_sizes([40])
            
            fig.suptitle(f"Approximated Contrast Distribution with training data ({X.shape[0]} points)")
            plt.tight_layout()
            plt.show()  # Now safe - running in main thread
            
            self.log_message("Distribution plot displayed successfully!")
            
        except Exception as e:
            self.log_message(f"Error creating distribution plot: {e}")
            import traceback
            traceback.print_exc()
    
    def show_help(self):
        """Show help dialog for the AMM training window"""
        show_help_dialog(self, "amm_training_dialog", "AMM Training Help")
    
    def keyPressEvent(self, event):
        """Handle keyboard shortcuts"""
        if event.key() == Qt.Key_F1:
            self.show_help()
        else:
            super().keyPressEvent(event)