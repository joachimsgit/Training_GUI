import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, title="Plot"):
        # Use smaller figure size and adjust for better spacing
        self.fig = Figure(figsize=(4, 3), dpi=80)
        super().__init__(self.fig)
        self.setParent(parent)
        
        # Create initial subplot with adjusted spacing
        self.axes = self.fig.add_subplot(111)
        self.axes.set_title(title, fontsize=10)
        self.axes.grid(True, alpha=0.3)
        
        # Improve spacing for axis labels
        self.fig.subplots_adjust(left=0.15, bottom=0.15, right=0.85, top=0.85)
        
        # Set background color
        self.fig.patch.set_facecolor('white')
        self.setStyleSheet("background-color: white;")
        
    def clear_plot(self):
        """Clear the plot while preserving title and grid"""
        title = self.axes.get_title()
        self.axes.clear()
        self.axes.set_title(title, fontsize=10)
        self.axes.grid(True, alpha=0.3)
        # Ensure proper spacing after clearing
        self.fig.subplots_adjust(left=0.15, bottom=0.15, right=0.85, top=0.85)
        self.draw()