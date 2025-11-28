"""
Help texts for all windows and dialogs in the GMM Training GUI application.
This file contains comprehensive documentation <h3>🎭 Mask Requirements:</h3>
<h4>🏷️ Required: Semantic Masks</h4>
<ul>
<li><b>Source</b>: Created in Class Annotator by converting instance masks</li>
<li><b>Format</b>: Multi-class PNG files (0=background, 1=class1, 2=class2, etc.)</li>
<li><b>Location</b>: [project]/semantic_masks/ folder</li>
<li><b>Auto-conversion</b>: Automatically converted to COCO format during training</li>
<li><b>Why Semantic Masks?</b>: M2F requires class-labeled data for training, instance masks must be converted first</li>
</ul>

<h4>🔄 COCO Format Conversion:</h4>
<ul>
<li><b>Process</b>: Semantic masks → COCO annotations with polygons/RLE encoding</li>
<li><b>Categories</b>: Multiple material classes from semantic masks</li>
<li><b>Output</b>: Standard COCO JSON format for Detectron2 training</li>
</ul>

<h3>📋 Training Workflow:</h3>
"""

HELP_TEXTS = {
    "main_window": """
<h2>📸 Main Window - Image Viewer and Analysis</h2>

<h3>🎯 Purpose:</h3>
<p>This is the main interface for viewing images, creating instance masks, and analyzing material contrast data.</p>

<h3>📋 Workflow:</h3>
<ol>
<li><b>Select Project Folder</b> - Choose a folder containing a folder with your material images</li>
<li><b>Navigate Images</b> - Use Previous/Next buttons or A/D keys</li>
<li><b>Create Instance Masks</b> - Click on the image to mark foreground/background regions</li>
<li><b>Save Instance Masks</b> - Press S or click Save Mask button</li>
<li><b>Analyze Data</b> - View contrast plots on the right panel</li>
</ol>

<h3>🎭 Mask Types in This Application:</h3>

<h4>📦 Instance Masks (Created HERE in Main Window):</h4>
<ul>
<li><b>Purpose</b>: Multi-valued masks with different values for each separate instance/object</li>
<li><b>Format</b>: Grayscale PNG with unique values for each instance (0=background, 1=instance1, 2=instance2, etc.)</li>
<li><b>Creation</b>: Click-based marking with watershed algorithm segmentation</li>
<li><b>Location</b>: Saved in [project]/masks/ folder as [image_name].png</li>
<li><b>Used For</b>: Must be converted to semantic masks before training (not directly used for AMM)</li>
</ul>

<h4>🏷️ Semantic Masks (Created in Class Annotator):</h4>
<ul>
<li><b>Purpose</b>: Multi-class masks with specific material classes (thicknesses)</li>
<li><b>Format</b>: Grayscale with different values for each class (0=background, 1=class1, 2=class2, etc.)</li>
<li><b>Creation</b>: Lasso selection on contrast plots to define class regions</li>
<li><b>Location</b>: Saved in [project]/semantic_masks/ folder</li>
<li><b>Used For</b>: AMM training, advanced analysis, multi-class training, detailed material classification</li>
</ul>

<h3>🔄 Mask Workflow:</h3>
<p><b>Step 1</b>: Create instance masks here → <b>Step 2</b>: Convert to semantic masks in Class Annotator → <b>Step 3</b>: Train models with semantic masks</p>

<h3>⌨️ Keyboard Shortcuts:</h3>
<ul>
<li><b>S</b> - Save current instance mask</li>
<li><b>A</b> - Previous image</li>
<li><b>D</b> - Next image</li>
<li><b>C</b> - Clear markers</li>
<li><b>M</b> - Toggle mask view</li>
</ul>

<h3>🎨 Instance Mask Creation:</h3>
<ul>
<li><b>Left Click</b> - Mark foreground (material)</li>
<li><b>Right Click</b> - Mark background</li>
<li><b>Green markers</b> - Foreground regions</li>
<li><b>Red markers</b> - Background regions</li>
<li><b>Watershed Algorithm</b> - Creates separate instances with unique values</li>
</ul>

<h3>📊 Plot Panel:</h3>
<p>Shows real-time contrast analysis between RGB channels. Toggle between cumulative data (all images) and single image view.</p>

<h3>🚀 Training Options:</h3>
<ul>
<li><b>Train AMM Model</b> - Uses semantic masks (converted from instance masks) for classification</li>
<li><b>Train M2F Model</b> - Uses semantic masks for deep learning segmentation</li>
<li><b>Annotate Classes</b> - Convert instance masks to semantic masks for training</li>
</ul>
""",

    "amm_training_dialog": """
<h2>🧠 AMM Training Dialog - Arbitrary Mixture Model</h2>

<h3>🎯 Purpose:</h3>
<p>Train an Arbitrary Mixture Model (AMM) for binary material/background classification using statistical analysis.</p>

<h3>🎭 Mask Requirements:</h3>
<h4>🏷️ Required: Semantic Masks</h4>
<ul>
<li><b>Source</b>: Created in Class Annotator by converting instance masks</li>
<li><b>Format</b>: Multi-class PNG files (0=background, 1=class1, 2=class2, etc.)</li>
<li><b>Location</b>: [project]/semantic_masks/ folder</li>
<li><b>Purpose</b>: Training data for class-based classification</li>
<li><b>Why Semantic Masks?</b>: AMM needs class-based data, instance masks must be converted first</li>
</ul>

<h3>📋 Training Process:</h3>
<ol>
<li><b>Data Preparation</b> - Loads semantic masks from Class Annotator</li>
<li><b>Feature Extraction</b> - Extracts RGB contrast features from class-labeled regions</li>
<li><b>Model Training</b> - Fits Arbitrary Mixture Model to distinguish material classes</li>
<li><b>Validation</b> - Tests on validation set</li>
<li><b>Model Saving</b> - Saves trained parameters for future use</li>
</ol>

<h3>⚙️ Parameters:</h3>
<ul>
<li><b>Number of Components</b> - How many distributions to fit</li>
<li><b>Random State</b> - Seed for reproducible results</li>
<li><b>Max Iterations</b> - Maximum training iterations</li>
<li><b>Tolerance</b> - Convergence threshold</li>
</ul>

<h3>📊 What it does:</h3>
<p>AMM analyzes the RGB contrast patterns in your semantic-masked regions and learns to distinguish between material classes based on color statistics. It's fast to train and run, making it ideal for real-time applications.</p>

<h3>💡 Best Practices:</h3>
<ul>
<li>Create instance masks in Main Window first, then convert to semantic masks in Class Annotator</li>
<li>Ensure you have semantic masks for diverse examples</li>
<li>Include various lighting conditions</li>
<li>Use 2-3 components for simple materials</li>
<li>Increase components for complex materials</li>
<li>At least 10-15 semantic masks recommended for good performance</li>
</ul>

<h3>📁 Output:</h3>
<p>Saves trained model parameters to the material folder for future material detection.</p>
""",

    "m2f_training_dialog": """
<h2>🔬 M2F Training Dialog - Deep Learning Segmentation</h2>

<h3>🎯 Purpose:</h3>
<p>Train a Mask2Former (M2F) deep learning model for high-precision image segmentation using advanced computer vision techniques.</p>

<h3>🎭 Mask Requirements:</h3>
<h4>� Required: Instance Masks</h4>
<ul>
<li><b>Source</b>: Created in Main Window using click-based marking</li>
<li><b>Format</b>: Binary PNG files (0=background, 255=material)</li>
<li><b>Location</b>: [project]/masks/ folder</li>
<li><b>Auto-conversion</b>: Automatically converted to COCO format during training</li>
<li><b>Why Instance Masks?</b>: M2F learns pixel-level segmentation, requiring precise boundary information from binary masks</li>
</ul>

<h4>🔄 COCO Format Conversion:</h4>
<ul>
<li><b>Process</b>: Instance masks → COCO annotations with polygons/RLE encoding</li>
<li><b>Categories</b>: Material (class 1) vs Background (class 0)</li>
<li><b>Output</b>: Standard COCO JSON format for Detectron2 training</li>
</ul>

<h3>�📋 Training Workflow:</h3>
<ol>
<li><b>Material Selection</b> - Choose which material to train on</li>
<li><b>Semantic Mask Loading</b> - Loads multi-class masks from [project]/semantic_masks/</li>
<li><b>COCO Conversion</b> - Converts semantic masks to COCO format automatically</li>
<li><b>Dataset Splitting</b> - Automatically splits into train/validation/test sets (image-level)</li>
<li><b>GPU Validation</b> - Checks available memory and recommends settings</li>
<li><b>Model Training</b> - Fine-tunes pre-trained Mask2Former model</li>
</ol>

<h3>🖥️ System Requirements:</h3>
<ul>
<li><b>GPU Memory</b>: 6GB+ recommended for optimal performance</li>
<li><b>4-6GB</b>: Reduce batch size to 1-2</li>
<li><b>2-4GB</b>: Use CPU training (slower but works)</li>
<li><b>&lt;2GB</b>: CPU training only</li>
</ul>

<h3>⚙️ Training Parameters:</h3>
<ul>
<li><b>Max Iterations</b> - Training duration (1000-5000 recommended)</li>
<li><b>Batch Size</b> - Images per batch (auto-adjusted for GPU)</li>
<li><b>Learning Rate</b> - Training speed (0.0001 default)</li>
<li><b>Device</b> - GPU (CUDA) or CPU training</li>
</ul>

<h3>📊 Data Management:</h3>
<ul>
<li><b>Auto-splitting</b>: 70% train, 20% validation, 10% test</li>
<li><b>Image-level separation</b>: Prevents data leakage between sets</li>
<li><b>COCO format</b>: Standard annotation format</li>
<li><b>Mask validation</b>: Checks semantic mask quality before training</li>
</ul>


<h3>💡 Training Tips:</h3>
<ul>
<li>Create instance masks in Main Window first, then convert to semantic masks in Class Annotator</li>
<li>At least 20-30 semantic masks recommended</li>
<li>Start with GPU memory check</li>
<li>Use default settings for first training</li>
<li>Monitor progress through tabs</li>
<li>Allow 30+ minutes for training</li>
</ul>

<h3>📁 Output Structure:</h3>
<p>Creates organized training directories within your material folder with COCO datasets, configs, and trained models.</p>
""",

    "class_annotator_dialog": """
<h2>🏷️ Class Annotator Dialog - Detailed Annotations</h2>

<h3>🎯 Purpose:</h3>
<p>Create semantic masks that are needed to train the AMM (Classification Model) and the M2F (Segmentation Model).</p>

<h3>🎭 Mask Creation:</h3>
<h4>🏷️ Creates: Semantic Masks</h4>
<ul>
<li><b>Purpose</b>: Multi-class classification of thickness (layers)</li>
<li><b>Format</b>: Grayscale PNG with pixel values representing classes (0=background, 1=class1, 2=class2, etc.)</li>
<li><b>Creation Method</b>: Lasso selection on RGB contrast plots to define class regions</li>
<li><b>Location</b>: Saved in [project]/semantic_masks/ folder as [image_name].png</li>
<li><b>Classes</b>: User-defined classes ("1-Layer", "2-Layer", etc.)</li>
</ul>

<h3>🔄 Prerequisite:</h3>
<ul>
<li><b>Required</b>: Instance masks must be created first in Main Window</li>
<li><b>Why?</b>: Contrast analysis needs material regions defined by instance masks</li>
<li><b>Process</b>: Instance masks → Contrast extraction → Class annotation → Semantic masks</li>
</ul>

<h3>📋 Annotation Workflow:</h3>
<ol>
<li><b>Load Contrast Data</b> - Uses instance masks to extract RGB contrast features</li>
<li><b>Select Class Number</b> - Choose current class (0-10) using spinbox or number keys</li>
<li><b>Lasso Selection</b> - Draw regions on contrast plots to define class areas</li>
<li><b>Assign Classes</b> - Regions get assigned to current class number</li>
<li><b>Save Semantic Masks</b> - Export multi-class masks for advanced analysis</li>
</ol>

<h3>🎨 Class Definition System:</h3>
<ul>
<li><b>Class 0</b>: Always background (automatically assigned)</li>
<li><b>Class 1-10</b>: User-defined material classes</li>
<li><b>Color Coding</b>: Each class gets unique color in visualization</li>
<li><b>Flexible</b>: Define classes based on your material properties</li>
</ul>

<h3>⌨️ Keyboard Shortcuts:</h3>
<ul>
<li><b>F1</b> - Show help</li>
<li><b>0-9</b> - Quick class selection</li>
<li><b>Mouse</b> - Lasso selection on plots</li>
</ul>


<h3>🔀 Mask Type Comparison:</h3>
<table border="1" style="border-collapse:collapse; margin:10px 0;">
<tr><th>Aspect</th><th>Instance Masks (Main Window)</th><th>Semantic Masks (Class Annotator)</th></tr>
<tr><td><b>Purpose</b></td><td>Multi-valued instance detection</td><td>Multi-class material classification</td></tr>
<tr><td><b>Classes</b></td><td>Multiple instances (0=background, 1=instance1, 2=instance2...)</td><td>Multiple classes (0=background, 1=class1, 2=class2...)</td></tr>
<tr><td><b>Creation</b></td><td>Click-based marking + Watershed algorithm</td><td>Lasso selection on contrast plots</td></tr>
<tr><td><b>File Format</b></td><td>Grayscale PNG (0, 1, 2, 3...)</td><td>Grayscale PNG (0, 1, 2, 3...)</td></tr>
<tr><td><b>File Name</b></td><td>[image_name].png</td><td>Various naming in semantic_masks/</td></tr>
<tr><td><b>Used For</b></td><td>Must be converted to semantic masks first</td><td>AMM training, M2F training, analysis</td></tr>
</table>

<h3>📁 Output:</h3>
<p>Saves detailed semantic mask files in [project]/semantic_masks/ compatible with advanced analysis tools and multi-class machine learning frameworks.</p>
""",

    "help_dialog": """
<h2>❓ Help System</h2>

<h3>🎯 Purpose:</h3>
<p>This help system provides comprehensive guidance for using the GMM Training GUI application.</p>

<h3>📋 Available Help:</h3>
<ul>
<li><b>Main Window</b> - Image viewing and mask creation</li>
<li><b>AMM Training</b> - Statistical model training</li>
<li><b>M2F Training</b> - Deep learning model training</li>
<li><b>Class Annotator</b> - Detailed annotations</li>
</ul>

<h3>💡 Getting Started:</h3>
<ol>
<li>Click the ❓ button in any window</li>
<li>Read the specific help for that window</li>
<li>Follow the workflow steps</li>
<li>Use keyboard shortcuts for efficiency</li>
</ol>

<h3>🚀 Tips:</h3>
<ul>
<li>Always start with the main window</li>
<li>Create good masks before training</li>
<li>Check system requirements for M2F</li>
<li>Use AMM for quick prototyping</li>
</ul>
"""
}


def get_help_text(window_name):
    """
    Get help text for a specific window.
    
    Args:
        window_name (str): Name of the window/dialog
        
    Returns:
        str: HTML formatted help text
    """
    return HELP_TEXTS.get(window_name, HELP_TEXTS["help_dialog"])


def get_all_help_topics():
    """
    Get list of all available help topics.
    
    Returns:
        list: List of available help topic names
    """
    return list(HELP_TEXTS.keys())