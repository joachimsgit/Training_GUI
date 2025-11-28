"""
Reusable help dialog component for displaying help texts.
"""

from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QTextEdit, QLabel)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from help_texts import get_help_text


class HelpDialog(QDialog):
    """
    A reusable help dialog that displays HTML-formatted help text.
    """
    
    def __init__(self, parent=None, window_name="help_dialog", title="Help"):
        super().__init__(parent)
        self.window_name = window_name
        self.setWindowTitle(title)
        self.setModal(True)
        self.resize(700, 600)
        
        self.init_ui()
        self.load_help_content()
    
    def init_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout(self)
        
        # Header
        header_layout = QHBoxLayout()
        
        # Title
        title_label = QLabel("📖 Help & Documentation")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title_label.setFont(title_font)
        header_layout.addWidget(title_label)
        
        header_layout.addStretch()
        
        # Close button
        close_btn = QPushButton("❌")
        close_btn.setFixedSize(30, 30)
        close_btn.clicked.connect(self.close)
        close_btn.setToolTip("Close Help")
        header_layout.addWidget(close_btn)
        
        layout.addLayout(header_layout)
        
        # Help content area
        self.help_text = QTextEdit()
        self.help_text.setReadOnly(True)
        self.help_text.setMinimumHeight(500)
        
        # Set font for better readability
        font = QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(10)
        self.help_text.setFont(font)
        
        layout.addWidget(self.help_text)
        
        # Bottom buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        ok_btn = QPushButton("OK")
        ok_btn.setMinimumWidth(100)
        ok_btn.clicked.connect(self.close)
        ok_btn.setDefault(True)
        button_layout.addWidget(ok_btn)
        
        layout.addLayout(button_layout)
    
    def load_help_content(self):
        """Load help content for the specified window"""
        help_content = get_help_text(self.window_name)
        self.help_text.setHtml(help_content)
        
        # Scroll to top
        cursor = self.help_text.textCursor()
        cursor.movePosition(cursor.Start)
        self.help_text.setTextCursor(cursor)


def show_help_dialog(parent, window_name, title="Help"):
    """
    Convenience function to show a help dialog.
    
    Args:
        parent: Parent widget
        window_name (str): Name of the window to get help for
        title (str): Dialog title
    """
    dialog = HelpDialog(parent, window_name, title)
    dialog.exec_()