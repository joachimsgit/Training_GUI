"""
Help texts for all windows and dialogs in the Training GUI application.
"""

HELP_TEXTS = {
    "main_window": """
<h2>Main Window - Image Viewer and Analysis</h2>

<h3>Purpose:</h3>
<p>View images, create instance masks, and analyze material contrast data.</p>

<h3>Workflow:</h3>
<ol>
<li>Select a project folder containing material images</li>
<li>Navigate images using Previous/Next or A/D keys</li>
<li>Create instance masks by clicking foreground/background regions</li>
<li>Save masks with S or the Save Mask button</li>
<li>View contrast plots on the right panel</li>
</ol>

<h3>Mask Types:</h3>
<h4>Instance Masks (Created here):</h4>
<ul>
<li>Grayscale PNG with unique values per instance (0=background, 1,2,3...=instances)</li>
<li>Created via click-based marking with watershed segmentation</li>
<li>Saved in [project]/masks/</li>
<li>Used for GMM training</li>
<li>Must be converted to semantic masks before AMM training</li>
</ul>

<h4>Semantic Masks (Created in Class Annotator):</h4>
<ul>
<li>Multi-class masks with values per class (0=background, 1,2,3...=classes)</li>
<li>Created via lasso selection on contrast plots</li>
<li>Saved in [project]/semantic_masks/</li>
<li>Required for AMM and M2F training</li>
</ul>

<h3>Keyboard Shortcuts:</h3>
<ul>
<li><b>S</b> - Save mask</li>
<li><b>A/D</b> - Previous/Next image</li>
<li><b>C</b> - Clear markers</li>
<li><b>M</b> - Toggle mask view</li>
</ul>

<h3>Mask Creation:</h3>
<ul>
<li><b>Left Click</b> - Mark foreground (green)</li>
<li><b>Right Click</b> - Mark background (red)</li>
</ul>
""",

    "amm_training_dialog": """
<h2>AMM Training - Arbitrary Mixture Model</h2>

<h3>Purpose:</h3>
<p>Train a statistical model for material classification based on RGB contrast patterns.</p>

<h3>Requirements:</h3>
<ul>
<li>Semantic masks in [project]/semantic_masks/</li>
<li>At least 10-15 masks recommended</li>
</ul>

<h3>Parameters:</h3>
<ul>
<li><b>Number of Components</b> - Distributions to fit (2-3 for simple materials)</li>
<li><b>Random State</b> - Seed for reproducibility</li>
<li><b>Max Iterations</b> - Training limit</li>
<li><b>Tolerance</b> - Convergence threshold</li>
</ul>

<h3>Tips:</h3>
<ul>
<li>Create instance masks first, then convert in Class Annotator</li>
<li>Include diverse lighting conditions</li>
<li>Increase components for complex materials</li>
</ul>

<h3>Output:</h3>
<p>Model parameters saved to the material folder.</p>
""",

    "m2f_training_dialog": """
<h2>M2F Training - Deep Learning Segmentation</h2>

<h3>Purpose:</h3>
<p>Train a Mask2Former model for high-precision image segmentation.</p>

<h3>Requirements:</h3>
<ul>
<li>Semantic masks in [project]/semantic_masks/</li>
<li>At least 20-30 masks recommended</li>
<li>GPU with 6GB+ memory (or CPU for slower training)</li>
</ul>

<h3>System Requirements:</h3>
<ul>
<li><b>6GB+ GPU</b> - Optimal performance</li>
<li><b>4-6GB GPU</b> - Reduce batch size to 1-2</li>
<li><b>2-4GB GPU</b> - Use CPU training</li>
</ul>

<h3>Parameters:</h3>
<ul>
<li><b>Max Iterations</b> - 1000-5000 recommended</li>
<li><b>Batch Size</b> - Auto-adjusted for GPU memory</li>
<li><b>Learning Rate</b> - Default 0.0001</li>
<li><b>Device</b> - GPU (CUDA) or CPU</li>
</ul>

<h3>Data Handling:</h3>
<ul>
<li>Auto-splits: 70% train, 20% validation, 10% test</li>
<li>Converts semantic masks to COCO format automatically</li>
</ul>

<h3>Tips:</h3>
<ul>
<li>Start with default settings</li>
<li>Allow 30+ minutes for training</li>
<li>Monitor progress through tabs</li>
</ul>
""",

    "class_annotator_dialog": """
<h2>Class Annotator - Semantic Mask Creation</h2>

<h3>Purpose:</h3>
<p>Convert instance masks to semantic masks for AMM and M2F training.</p>

<h3>Prerequisites:</h3>
<ul>
<li>Instance masks must exist in [project]/masks/</li>
</ul>

<h3>Workflow:</h3>
<ol>
<li>Load contrast data from instance masks</li>
<li>Select class number (0-10) using spinbox or number keys</li>
<li>Draw lasso selection on contrast plots</li>
<li>Save semantic masks</li>
</ol>

<h3>Class System:</h3>
<ul>
<li><b>Class 0</b> - Background (automatic)</li>
<li><b>Class 1-10</b> - User-defined material classes</li>
</ul>

<h3>Keyboard Shortcuts:</h3>
<ul>
<li><b>F1</b> - Show help</li>
<li><b>0-9</b> - Quick class selection</li>
</ul>

<h3>Output:</h3>
<p>Semantic masks saved in [project]/semantic_masks/</p>
""",

    "help_dialog": """
<h2>Help System</h2>

<h3>Available Topics:</h3>
<ul>
<li><b>Main Window</b> - Image viewing and mask creation</li>
<li><b>AMM Training</b> - Statistical model training</li>
<li><b>M2F Training</b> - Deep learning model training</li>
<li><b>Class Annotator</b> - Semantic mask creation</li>
</ul>

<h3>Quick Start:</h3>
<ol>
<li>Create instance masks in Main Window</li>
<li>Convert to semantic masks in Class Annotator</li>
<li>Train models (AMM for speed, M2F for accuracy)</li>
</ol>
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