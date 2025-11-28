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


class AMMTrainingThread(QThread):
    """Separate thread for AMM training to prevent GUI freezing"""
    progress_update = pyqtSignal(int)  # Progress percentage
    log_message = pyqtSignal(str)  # Log messages
    training_complete = pyqtSignal(bool)  # Success/failure
    
    def __init__(self, project_folder, training_params):
        super().__init__()
        self.project_folder = project_folder
        self.training_params = training_params
        self.should_stop = False
        
    def run(self):
        """Run the AMM training process"""
        try:
            self.log_message.emit("🚀 Starting AMM training...")
            
            # Add maskterial to path
            maskterial_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "maskterial")
            if maskterial_path not in sys.path:
                sys.path.append(maskterial_path)
                
            from maskterial.modeling.common.fcresnet import FCResNet
            
            # Set up data paths
            images_dir = os.path.join(self.project_folder, "images")
            semantic_masks_dir = os.path.join(self.project_folder, "semantic_masks")
            
            self.progress_update.emit(10)
            self.log_message.emit("📂 Loading training data...")
            
            # Load data using our custom function instead of maskterial's loader
            X_train, y_train, num_classes = self.load_contrast_data_from_gui_format(
                images_dir, semantic_masks_dir
            )
            
            if len(X_train) == 0:
                self.log_message.emit("❌ No training data found!")
                self.training_complete.emit(False)
                return
            
            self.progress_update.emit(30)
            self.log_message.emit(f"✅ Loaded {len(X_train)} training samples from {num_classes} classes")
            
            if self.should_stop:
                return
                
            # Create model
            model = FCResNet(
                input_dim=self.training_params.get('input_dim', 3),
                num_classes=num_classes,
                embedding_dim=self.training_params.get('embedding_dim', 16),
                depth=self.training_params.get('depth', 4),
                spectral_normalization=self.training_params.get('spectral_normalization', True),
                spec_coeff=self.training_params.get('spec_coeff', 0.5),
                n_power_iterations=self.training_params.get('n_power_iterations', 5),
                dropout_rate=self.training_params.get('dropout_rate', 0.1)
            )
            
            self.progress_update.emit(40)
            self.log_message.emit("🧠 Model created, starting training...")
            
            # Set up training
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=self.training_params.get('learning_rate', 0.01))
            
            # Convert data to tensors
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
            y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
            
            # Training loop
            num_iterations = self.training_params.get('num_iterations', 5000)
            batch_size = self.training_params.get('batch_size', 10000)
            
            model.train()
            for iteration in range(num_iterations):
                if self.should_stop:
                    break
                    
                # Random batch sampling
                indices = torch.randperm(len(X_train_tensor))[:batch_size]
                batch_X = X_train_tensor[indices]
                batch_y = y_train_tensor[indices]
                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Progress update
                progress = 40 + int((iteration / num_iterations) * 50)
                self.progress_update.emit(progress)
                
                # Log every 100 iterations
                if iteration % 100 == 0:
                    accuracy = self.calculate_accuracy(model, X_train_tensor, y_train_tensor, device)
                    self.log_message.emit(f"Iteration {iteration}/{num_iterations}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%")
            
            self.progress_update.emit(90)
            self.log_message.emit("🧮 Calculating class embeddings for AMM...")
            
            # Calculate class embeddings (GMM parameters) - THIS IS CRITICAL!
            loc, cov = self.calculate_class_embeddings(model, X_train_tensor, y_train_tensor, num_classes, device)
            
            self.progress_update.emit(95)
            self.log_message.emit("💾 Saving model and AMM parameters...")
            
            # Save model and AMM parameters
            model_dir = os.path.join(self.project_folder, "amm_model")
            os.makedirs(model_dir, exist_ok=True)
            
            # Save PyTorch model
            model_path = os.path.join(model_dir, "model.pth")
            torch.save(model.state_dict(), model_path)
            
            # Save AMM parameters (critical for inference)
            np.save(os.path.join(model_dir, "loc.npy"), loc.cpu().numpy())
            np.save(os.path.join(model_dir, "cov.npy"), cov.cpu().numpy())
            
            # Save metadata
            meta_data = {
                'model_config': {
                    'input_dim': self.training_params.get('input_dim', 3),
                    'num_classes': num_classes,
                    'embedding_dim': self.training_params.get('embedding_dim', 16),
                    'depth': self.training_params.get('depth', 4),
                },
                'training_params': self.training_params,
                'num_classes': num_classes,
                'class_ids': list(range(num_classes)),
                'final_loss': loss.item(),
                'final_accuracy': accuracy
            }
            
            import json
            with open(os.path.join(model_dir, "meta_data.json"), 'w') as f:
                json.dump(meta_data, f, indent=4)
            
            self.progress_update.emit(100)
            self.log_message.emit(f"✅ Training completed! AMM model saved to: {model_dir}")
            self.log_message.emit(f"📁 Files created: model.pth, loc.npy, cov.npy, meta_data.json")
            self.training_complete.emit(True)
            
        except Exception as e:
            self.log_message.emit(f"❌ Training failed: {str(e)}")
            self.training_complete.emit(False)
            
    def calculate_accuracy(self, model, X, y, device, sample_size=1000):
        """Calculate accuracy on a sample of the data"""
        model.eval()
        with torch.no_grad():
            # Sample for efficiency
            indices = torch.randperm(len(X))[:sample_size]
            sample_X = X[indices]
            sample_y = y[indices]
            
            outputs = model(sample_X)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == sample_y).sum().item()
            accuracy = 100 * correct / sample_size
            
        model.train()
        return accuracy
        
    def stop(self):
        """Stop the training process"""
        self.should_stop = True
        
    def load_contrast_data_from_gui_format(self, images_dir, semantic_masks_dir):
        """Load contrast data from our GUI's format"""
        import cv2
        
        X_all = []
        y_all = []
        
        # Get list of semantic mask files - look for files with same names as images
        semantic_files = [f for f in os.listdir(semantic_masks_dir) 
                         if f.endswith('.png') and not f.endswith('_semantic.png')]
        
        if not semantic_files:
            # Fallback: try looking for _semantic suffix files
            semantic_files = [f for f in os.listdir(semantic_masks_dir) if f.endswith('_semantic.png')]
            use_semantic_suffix = True
        else:
            use_semantic_suffix = False
            
        if not semantic_files:
            return np.array([]), np.array([]), 0
            
        self.log_message.emit(f"Processing {len(semantic_files)} semantic masks...")
        
        for i, semantic_file in enumerate(semantic_files):
            # Find corresponding image file
            if use_semantic_suffix:
                # Old naming: semantic_file ends with _semantic.png
                image_name = semantic_file.replace('_semantic.png', '.png')
            else:
                # New naming: semantic_file has same name as image
                image_name = semantic_file
                
            image_path = os.path.join(images_dir, image_name)
            semantic_path = os.path.join(semantic_masks_dir, semantic_file)
            
            # Try different image extensions if exact match not found
            if not os.path.exists(image_path):
                base_name = os.path.splitext(image_name)[0]
                for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                    alt_image_path = os.path.join(images_dir, base_name + ext)
                    if os.path.exists(alt_image_path):
                        image_path = alt_image_path
                        break
                        
            if not os.path.exists(image_path):
                self.log_message.emit(f"⚠️ Image not found for {semantic_file}, skipping...")
                continue
                
            # Load image and semantic mask
            image = cv2.imread(image_path)
            semantic_mask = cv2.imread(semantic_path, 0)  # Grayscale
            
            if image is None or semantic_mask is None:
                self.log_message.emit(f"⚠️ Failed to load {semantic_file}, skipping...")
                continue
                
            # Calculate background color (mean of background pixels - class 0)
            background_pixels = semantic_mask == 0
            if np.any(background_pixels):
                bg_b_mean = np.mean(image[background_pixels, 0])  # Blue channel
                bg_g_mean = np.mean(image[background_pixels, 1])  # Green channel
                bg_r_mean = np.mean(image[background_pixels, 2])  # Red channel
                
                # Avoid division by zero
                bg_b_mean = max(bg_b_mean, 1)
                bg_g_mean = max(bg_g_mean, 1)
                bg_r_mean = max(bg_r_mean, 1)
                
                background_color = np.array([bg_b_mean, bg_g_mean, bg_r_mean])
            else:
                # Fallback if no background found
                background_color = np.array([128, 128, 128])
                
            # Calculate contrast for each class
            unique_classes = np.unique(semantic_mask)
            for class_id in unique_classes:
                if class_id == 0:  # Skip background
                    continue
                    
                # Get pixels for this class
                class_pixels = semantic_mask == class_id
                if not np.any(class_pixels):
                    continue
                    
                # Extract RGB values and calculate contrast
                class_bgr = image[class_pixels]  # Shape: (n_pixels, 3)
                
                # Calculate contrast: (pixel_color / background_color) - 1
                # Convert BGR to RGB order for consistency
                class_rgb = class_bgr[:, [2, 1, 0]]  # BGR -> RGB
                background_rgb = background_color[[2, 1, 0]]  # BGR -> RGB
                
                contrast_values = (class_rgb / background_rgb) - 1
                
                # Add to training data
                X_all.extend(contrast_values)
                y_all.extend([class_id] * len(contrast_values))
                
            # Progress update
            progress = 10 + int((i / len(semantic_files)) * 20)
            self.progress_update.emit(progress)
            
        if not X_all:
            return np.array([]), np.array([]), 0
            
        X_all = np.array(X_all, dtype=np.float32)
        y_all = np.array(y_all, dtype=np.int64)
        
        # Remap class IDs to be contiguous starting from 0
        unique_classes = np.unique(y_all)
        class_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_classes)}
        y_all = np.array([class_mapping[class_id] for class_id in y_all])
        
        num_classes = len(unique_classes)
        
        self.log_message.emit(f"✅ Extracted {len(X_all)} contrast samples from {num_classes} classes")
        
        return X_all, y_all, num_classes
        
    def calculate_class_embeddings(self, model, X_train, y_train, num_classes, device):
        """
        Calculate mean and covariance matrices of embeddings for each class.
        This is the critical AMM component that creates the Gaussian mixture model.
        """
        model.eval()
        with torch.no_grad():
            # Get embeddings for all training data
            X_embeddings = model.get_embedding(X_train)
            
            # Calculate mean (loc) and covariance (cov) for each class
            loc = torch.stack([
                torch.mean(X_embeddings[y_train == c], dim=0)
                for c in range(num_classes)
            ])
            
            cov = torch.stack([
                torch.cov(X_embeddings[y_train == c].T)
                for c in range(num_classes)
            ])
            
        model.train()
        return loc, cov

class AMMTrainingDialog(QDialog):
    def __init__(self, parent=None, project_folder=None):
        super().__init__(parent)
        self.project_folder = project_folder
        self.training_thread = None
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize the minimal training UI"""
        self.setWindowTitle("AMM Training")
        self.resize(600, 400)
        
        layout = QVBoxLayout()
        
        # Status
        self.status_label = QLabel("Ready to train AMM model")
        layout.addWidget(self.status_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        layout.addWidget(self.progress_bar)
        
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
        
        # Training log (minimal)
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(200)
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)
        
        self.setLayout(layout)
        
        # Check if training data is available
        self.check_training_data()
        
    def check_training_data(self):
        """Quick check for training data availability"""
        if not self.project_folder:
            self.status_label.setText("No project folder specified")
            return
            
        semantic_masks_dir = os.path.join(self.project_folder, "semantic_masks")
        if os.path.exists(semantic_masks_dir):
            semantic_files = [f for f in os.listdir(semantic_masks_dir) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if semantic_files:
                self.status_label.setText(f"Ready - Found {len(semantic_files)} semantic masks")
                return
        
        self.status_label.setText("No semantic masks found - Please use Class Annotator first")
        self.btn_start.setEnabled(False)
        
    def start_training(self):
        """Start the AMM training process"""
        self.log_message("Starting AMM training...")
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress_bar.setValue(0)
        
        # Use fixed training parameters for simplicity
        training_params = {
            'num_iterations': 5000,
            'learning_rate': 0.01,
            'batch_size': 10000,
            'input_dim': 3,
            'embedding_dim': 16,
            'depth': 4,
            'spectral_normalization': True,
            'spec_coeff': 0.5,
            'n_power_iterations': 5,
            'dropout_rate': 0.1
        }
        
        # Start training thread
        self.training_thread = AMMTrainingThread(self.project_folder, training_params)
        self.training_thread.progress_update.connect(self.update_progress)
        self.training_thread.log_message.connect(self.log_message)
        self.training_thread.training_complete.connect(self.on_training_complete)
        self.training_thread.start()
        
    def update_progress(self, value):
        """Update progress bar"""
        self.progress_bar.setValue(value)
        
    def on_training_complete(self, success):
        """Handle training completion"""
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        
        if success:
            self.log_message("Training completed successfully!")
            self.status_label.setText("Training completed - AMM model saved")
        else:
            self.log_message("Training failed. Check the log for details.")
            self.status_label.setText("Training failed")
        
    def stop_training(self):
        """Stop the training process"""
        self.log_message("Stopping training...")
        if self.training_thread and self.training_thread.isRunning():
            self.training_thread.should_stop = True
            self.training_thread.wait()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
            
    def log_message(self, message):
        """Add a message to the training log"""
        self.log_text.append(message)
        self.log_text.ensureCursorVisible()