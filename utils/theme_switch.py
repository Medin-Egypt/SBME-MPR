from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import Qt, pyqtSignal, QRect, QPropertyAnimation, QEasingCurve, pyqtProperty
from PyQt5.QtGui import QPainter, QColor, QPixmap, QIcon, QPainterPath


class ThemeSwitchWidget(QWidget):
    """Custom widget that displays a theme switch with dark/light mode icons and animated transition"""
    
    theme_changed = pyqtSignal(bool)  # True = dark mode, False = light mode
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.is_dark_mode = True  # Default to dark mode
        
        # Load icons
        self.dark_icon = QIcon("Icons/night-mode.png")
        self.light_icon = QIcon("Icons/light-mode.png")
        
        # Set fixed size
        self.setFixedSize(80, 30)
        self.setToolTip("Switch Theme")
        
        # Make it clickable
        self.setCursor(Qt.PointingHandCursor)
        
        # Animation property for smooth sliding
        self._slide_position = 0.0  # 0.0 = full left (dark), 1.0 = full right (light)
        
        # Create animation
        self.animation = QPropertyAnimation(self, b"slidePosition")
        self.animation.setDuration(300)  # 300ms animation
        self.animation.setEasingCurve(QEasingCurve.InOutCubic)
    
    @pyqtProperty(float)
    def slidePosition(self):
        return self._slide_position
    
    @slidePosition.setter
    def slidePosition(self, value):
        self._slide_position = value
        self.update()  # Trigger repaint
    
    def paintEvent(self, event):
        """Custom paint to draw the switch with rounded corners"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Define colors based on current theme
        if self.is_dark_mode:
            bg_color = QColor("#2D3748")
            active_bg = QColor("#4A5568")
            separator_color = QColor("#63B3ED")
        else:
            bg_color = QColor("#E2E8F0")
            active_bg = QColor("#CBD5E0")
            separator_color = QColor("#4299E1")
        
        # Create rounded rectangle path for background
        path = QPainterPath()
        path.addRoundedRect(0, 0, 80, 30, 15, 15)  # 15px radius for rounded corners
        
        # Draw background with rounded corners
        painter.setClipPath(path)
        painter.fillRect(self.rect(), bg_color)
        
        # Calculate animated highlight position
        # slide_position: 0.0 = left (dark), 1.0 = right (light)
        highlight_x = int(40 * self._slide_position)
        highlight_width = 40
        
        # Draw animated active side highlight
        painter.fillRect(QRect(highlight_x, 0, highlight_width, 30), active_bg)
        
        # Draw vertical separator line in the middle
        painter.setPen(separator_color)
        painter.drawLine(40, 5, 40, 25)
        
        # Draw icons
        icon_size = 20
        icon_y = (30 - icon_size) // 2
        
        # Dark mode icon (left)
        dark_pixmap = self.dark_icon.pixmap(icon_size, icon_size)
        painter.drawPixmap(10, icon_y, dark_pixmap)
        
        # Light mode icon (right)
        light_pixmap = self.light_icon.pixmap(icon_size, icon_size)
        painter.drawPixmap(50, icon_y, light_pixmap)
        
        painter.end()
    
    def mousePressEvent(self, event):
        """Handle mouse click to switch theme with animation"""
        if event.button() == Qt.LeftButton:
            # Determine which side was clicked
            if event.x() < 40:
                # Left side clicked - switch to dark mode
                if not self.is_dark_mode:
                    self.is_dark_mode = True
                    self._animate_to_dark()
                    self.theme_changed.emit(True)
            else:
                # Right side clicked - switch to light mode
                if self.is_dark_mode:
                    self.is_dark_mode = False
                    self._animate_to_light()
                    self.theme_changed.emit(False)
    
    def _animate_to_dark(self):
        """Animate highlight sliding to dark mode (left)"""
        self.animation.stop()
        self.animation.setStartValue(self._slide_position)
        self.animation.setEndValue(0.0)
        self.animation.start()
    
    def _animate_to_light(self):
        """Animate highlight sliding to light mode (right)"""
        self.animation.stop()
        self.animation.setStartValue(self._slide_position)
        self.animation.setEndValue(1.0)
        self.animation.start()