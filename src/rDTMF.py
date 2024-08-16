import sys
import os
import time
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import (QApplication, QMainWindow, QComboBox, QPushButton, QVBoxLayout, QWidget, QLabel, QTextEdit, QHBoxLayout, 
                             QMenuBar, QAction, QDialog, QFormLayout, QLineEdit, QPushButton as QDialogButton)
from PyQt5.QtCore import QThread, pyqtSignal, QObject, Qt, QPoint, QTimer
from PyQt5.QtGui import QPainter, QColor
import sounddevice as sd
from scipy import signal
import numpy as np

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS2
        base_path = sys._MEIPASS2
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

class DTMFDetector(QObject):
    dtmf_detected = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.dtmf_freqs = {
            '1': (697, 1209), '2': (697, 1336), '3': (697, 1477), 'A': (697, 1633),
            '4': (770, 1209), '5': (770, 1336), '6': (770, 1477), 'B': (770, 1633),
            '7': (852, 1209), '8': (852, 1336), '9': (852, 1477), 'C': (852, 1633),
            '*': (941, 1209), '0': (941, 1336), '#': (941, 1477), 'D': (941, 1633)
        }
        self.delay = 0.2  # Default delay in seconds
        self.last_detection_time = time.time()
        self.detection_queue = []

    def detect_dtmf(self, audio_data):
        self.detection_queue.append(audio_data)

    def process_queue(self):
        current_time = time.time()
        if current_time - self.last_detection_time >= self.delay and self.detection_queue:
            audio_data = self.detection_queue.pop(0)
            self.detect_from_data(audio_data)
            self.last_detection_time = current_time

    def detect_from_data(self, audio_data):
        if len(audio_data) == 0:
            return

        fft = np.fft.rfft(audio_data[:, 0])
        freq = np.fft.rfftfreq(len(audio_data), d=1/44100)
        magnitudes = np.abs(fft)

        peak_height = np.max(magnitudes) * 0.68
        peaks, _ = signal.find_peaks(magnitudes, height=peak_height, distance=2)
        detected_freqs = freq[peaks][magnitudes[peaks] > 13]

        valid_range = (600, 1700)
        detected_freqs = detected_freqs[(detected_freqs >= valid_range[0]) & (detected_freqs <= valid_range[1])]

        if len(detected_freqs) < 2:
            return

        detected_freqs = sorted(detected_freqs)
        possible_pairs = []
        for i in range(len(detected_freqs) - 1):
            possible_pairs.append((detected_freqs[i], detected_freqs[i + 1]))

        best_match = None
        min_distance = float('inf')
        for key, (low_freq, high_freq) in self.dtmf_freqs.items():
            for pair in possible_pairs:
                distance = abs(pair[0] - low_freq) + abs(pair[1] - high_freq)
                if distance < min_distance:
                    min_distance = distance
                    best_match = key

        if best_match:
            self.dtmf_detected.emit(best_match)

class AudioVisualizer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.level = 0  # Current sound level
        self.smoothed_level = 0  # Smoothed sound level
        self.color = QColor('#A594F9')  # Color for the meter
        self.setFixedWidth(50)  # Set a fixed width for the visualizer
        self.setMinimumHeight(200)  # Ensure the visualizer has a minimum height
        self.smoothing_factor = 0.25  # Smoothing factor for the exponential moving average

    def update_bars(self, audio_data):
        if len(audio_data) == 0:
            # If no audio data, smoothly transition to 0
            target_level = 0
        else:
            # Compute FFT and get magnitudes
            fft = np.fft.rfft(audio_data[:, 0])
            magnitudes = np.abs(fft)
            
            # Get the maximum magnitude and normalize for the visualizer
            max_magnitude = np.max(magnitudes)
            if max_magnitude == 0:  # Avoid division by zero
                max_magnitude = 1
            target_level = np.interp(max_magnitude, (0, 100), (0, self.height()))  # Scale the level

        # Apply exponential moving average for smoothing
        self.smoothed_level = (self.smoothing_factor * target_level) + ((1 - self.smoothing_factor) * self.smoothed_level)

        self.update()  # Request a repaint

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setBrush(self.color)
        
        # Draw a single vertical bar representing the smoothed sound level
        bar_width = self.width()
        height = int(self.smoothed_level)
        painter.drawRect(0, self.height() - height, bar_width, height)

class AudioThread(QThread):
    audio_data_signal = pyqtSignal(np.ndarray)
    dtmf_detected = pyqtSignal(str)
    
    def __init__(self, input_device, output_device, visualizer=None, main_window=None):
        super().__init__()
        self.input_device = input_device
        self.output_device = output_device
        self.visualizer = visualizer
        self.main_window = main_window  # Reference to MainWindow instance
        self.running = False
        self.buffer_size = 2048
        self.sample_rate = 44100
        self.detector = DTMFDetector()
        self.detector.dtmf_detected.connect(self.handle_dtmf_detected)

    def run(self):
        self.running = True
        try:
            with sd.Stream(device=(self.input_device, self.output_device),
                           samplerate=self.sample_rate, 
                           blocksize=self.buffer_size,
                           channels=1, 
                           callback=self.audio_callback):
                while self.running:
                    sd.sleep(100)
                    self.detector.process_queue()
        except sd.PortAudioError as e:
            print(f"Audio error: {e}")

    def audio_callback(self, indata, outdata, frames, time, status):
        if status:
            print(f"Status: {status}")
        if self.main_window and getattr(self.main_window, 'stop_transition', False):
            # Smooth transition to zero if stop_transition is True
            self.visualizer.update_bars(np.zeros_like(indata))
            outdata[:] = np.zeros_like(outdata)  # Ensure output is zero
        else:
            outdata[:] = indata
            self.audio_data_signal.emit(indata)
            if self.visualizer:
                self.visualizer.update_bars(indata)
            self.detector.detect_dtmf(indata)

    def handle_dtmf_detected(self, dtmf_char):
        self.dtmf_detected.emit(dtmf_char)

    def stop(self):
        self.running = False  # Set running flag to False to stop the thread
        # Additional clean-up code if needed
        self.wait()  # Wait for the thread to finish execution

class SettingsDialog(QDialog):
    def __init__(self, title, setting_name, current_value, callback, parent=None):
        super().__init__(parent)
        self.callback = callback
        self.setWindowTitle(title)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)  # Remove '?' icon
        self.setStyleSheet("background-color: #051014; color: #FB8B24;")
        layout = QFormLayout(self)
        self.value_input = QLineEdit()
        self.value_input.setText(str(current_value))
        layout.addRow(f"{setting_name}:", self.value_input)
        ok_button = QDialogButton("OK")
        ok_button.setStyleSheet("background-color: #555287; color: #FB8B24;")
        ok_button.clicked.connect(self.accept)
        cancel_button = QDialogButton("Cancel")
        cancel_button.setStyleSheet("background-color: #555287; color: #FB8B24;")
        cancel_button.clicked.connect(self.reject)
        layout.addRow(ok_button, cancel_button)

    def accept(self):
        try:
            value = float(self.value_input.text())
            self.callback(value)
            super().accept()
        except ValueError:
            self.value_input.setText("Invalid input")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # self.setWindowIcon(QtGui.QIcon('windico.png'))
        self.setWindowTitle("rDTMF")
        self.setFixedSize(600, 400)
        self.stop_transition = False
        
        # Timer for smooth transition
        self.transition_timer = QTimer()
        self.transition_timer.timeout.connect(self.update_visualizer_transition)
        self.transition_timer.start(75)  # Timer interval in milliseconds

        # Set the main window background color
        self.setStyleSheet("background-color: #051014;")

        # Create the visualizer
        self.visualizer = AudioVisualizer()

        # Widgets for input and output
        self.input_label = QLabel("Select Input Device:")
        self.input_combo = QComboBox()
        self.output_label = QLabel("Select Output Device:")
        self.output_combo = QComboBox()
        self.start_button = QPushButton("Start Playback && Decoding")
        self.stop_button = QPushButton("Stop")
        self.dtmf_output = QTextEdit()
        self.dtmf_output.setReadOnly(True)  # Disable text input
        self.clear_button = QPushButton("Clear All Text")
        self.clear_button.setFixedSize(100,30);


        # Style the buttons and the decoded text field 
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #555287;
                color: #FB8B24;                        
                font-size: 18px;
                border-style: outset;
                border-width: 2px;
                border-radius: 4px;
                border-color: #051014;
                padding: 4px;                        
            }
            QPushButton:disabled {
                background-color: #1A2022;                        
            }
        """)
        self.stop_button.setStyleSheet("""
            QPushButton {
                background-color: #555287;
                color: #FB8B24;                        
                font-size: 18px;
                border-style: outset;
                border-width: 2px;
                border-radius: 4px;
                border-color: #051014;
                padding: 4px;                        
            }
            QPushButton:disabled {
                background-color: #1A2022;
            }
        """)
        
        self.dtmf_output.setStyleSheet("""
            QTextEdit {
                background-color: #2E2F2F;
                color: #D7263D;
                selection-background-color: #555287;
                border: 1px solid #555287;  
            }
            QTextEdit QScrollBar:vertical {
                background: #2E2F2F;
                border: 2px solid #555287;
                width: 12px;
                
            }
            QTextEdit QScrollBar::handle:vertical {
                background: #555287;
                border-radius: 6px;
            }
            QScrollBar::sub-page:vertical {
                border: none;                     
                background: none;
            }
            
            QScrollBar::add-page:vetrical {
                border: none;     
                background: none;
            }
            QTextEdit QScrollBar::up-arrow:vertical,
            QTextEdit QScrollBar::down-arrow:vertical {
                background: none;
                color: #555287;                       
            }
            QTextEdit QMenu {
                background-color: #051014;  /* Background color of the menu */
                color: #FB8B24;             /* Text color of the menu items */
            }
            QTextEdit QMenu::item:selected {
                background-color: #555287;  /* Background color of the selected item */
                color: #FB8B24;             /* Text color of the selected item */
            }
        """)
        
        self.clear_button.setStyleSheet("""
            QPushButton {
                background-color: #D7263D;
                color: #D9D9DD;                        
                font-size: 12px;
                border-style: outset;
                border-width: 2px;
                border-radius: 4px;
                border-color: #051014;
                padding: 4px;                        
            }
        """)
        
        # Style the text labels
        self.input_label.setStyleSheet("""
            QLabel {
                color: #C4B7CB;
            }
        """)
        self.output_label.setStyleSheet("""
            QLabel {
                color: #C4B7CB;
            }
        """)
        
        self.input_combo.setStyleSheet("""
            QComboBox {
                color: #FB8B24;
                selection-background-color: #555287;
            }
            QListView {
                color: #FB8B24;
                selection-background-color: #555287;                    
            }
        """)
        self.output_combo.setStyleSheet("""
            QComboBox {
                color: #FB8B24;
                selection-background-color: #555287;
            }
            QListView {
                color: #FB8B24;
                selection-background-color: #555287;                    
            }                            
        """)

        # Menu Bar
        self.menu_bar = self.menuBar()
        self.menu_bar.setStyleSheet("""
            QMenuBar {
                background-color: #555287;
                color: #051014;
                font-weight: bold;
            }
            QMenu {
                background-color: #051014;
                color: #FB8B24;
            }
            QMenu::item:selected {
                background-color: #555287;
            }
        """)

        # Options Menu
        options_menu = self.menu_bar.addMenu("Options")
        sample_rate_action = QAction("Sample Rate", self)
        sample_rate_action.triggered.connect(self.show_sample_rate_dialog)
        options_menu.addAction(sample_rate_action)
        
        timing_action = QAction("Timing (Delay)", self)
        timing_action.triggered.connect(self.show_timing_dialog)
        options_menu.addAction(timing_action)

        buffer_size_action = QAction("Buffer Size", self)
        buffer_size_action.triggered.connect(self.show_buffer_size_dialog)
        options_menu.addAction(buffer_size_action)

        # About Menu
        about_menu = self.menu_bar.addMenu("About")
        about_dtmf_action = QAction("About DTMF Decoder", self)
        about_dtmf_action.triggered.connect(self.show_about_dtmf_dialog)
        about_menu.addAction(about_dtmf_action)

        about_author_action = QAction("Made by RealRatnadwip", self)
        about_author_action.triggered.connect(self.open_author_link)
        about_menu.addAction(about_author_action)

        # Main layout
        main_layout = QHBoxLayout()

        # Create a layout for controls and place it to the right of the visualizer
        controls_layout = QVBoxLayout()
        controls_layout.addWidget(self.input_label)
        controls_layout.addWidget(self.input_combo)
        controls_layout.addWidget(self.output_label)
        controls_layout.addWidget(self.output_combo)
        controls_layout.addWidget(self.start_button)
        controls_layout.addWidget(self.stop_button)
        controls_layout.addWidget(self.dtmf_output)
        controls_layout.addWidget(self.clear_button)

        # Add visualizer and controls layout to the main layout
        main_layout.addWidget(self.visualizer)  # Add visualizer on the left
        main_layout.addLayout(controls_layout)  # Add controls on the right

        # Set the layout to the central widget
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        self.populate_devices()
        self.start_button.clicked.connect(self.start_detection)
        self.stop_button.clicked.connect(self.stop_detection)
        self.stop_button.setEnabled(False)
        self.clear_button.clicked.connect(self.clear_dtmf_output)

        self.audio_thread = None
        
        # Flag to indicate stopping
        stop_transition = False


    def populate_devices(self):
        devices = sd.query_devices()
        for device in devices:
            if device['max_input_channels'] > 0:
                self.input_combo.addItem(f"{device['name']} (Input)", device['index'])
            if device['max_output_channels'] > 0:
                self.output_combo.addItem(f"{device['name']} (Output)", device['index'])

    def start_detection(self):
        input_device = self.input_combo.currentData()
        output_device = self.output_combo.currentData()
        self.audio_thread = AudioThread(input_device, output_device, self.visualizer)
        self.audio_thread.dtmf_detected.connect(self.update_dtmf_label)

        self.audio_thread.start()
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        
        self.stop_transition = False  # Reset stop transition flag

    def stop_detection(self):
        if self.audio_thread:
            self.stop_transition = True
            self.audio_thread.stop()  # Stop the audio thread
            self.audio_thread = None
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        
    def update_visualizer_transition(self):
        if self.stop_transition and self.visualizer:
            # Smooth transition to zero
            self.visualizer.update_bars(np.zeros((self.visualizer.height(), 1)))

    def update_dtmf_label(self, dtmf_char):
        self.dtmf_output.append(dtmf_char)
        
    def clear_dtmf_output(self):
        self.dtmf_output.clear()

    def show_sample_rate_dialog(self):
        if self.audio_thread:
            current_rate = self.audio_thread.sample_rate
        else:
            current_rate = 44100
        dialog = SettingsDialog("Sample Rate", "Sample Rate (Hz)", current_rate, self.set_sample_rate, self)
        dialog.exec_()

    def show_timing_dialog(self):
        if self.audio_thread:
            current_delay = self.audio_thread.detector.delay
        else:
            current_delay = 0.2
        dialog = SettingsDialog("Timing (Delay)", "Delay (seconds)", current_delay, self.set_timing, self)
        dialog.exec_()

    def show_buffer_size_dialog(self):
        if self.audio_thread:
            current_buffer_size = self.audio_thread.buffer_size
        else:
            current_buffer_size = 2048
        dialog = SettingsDialog("Buffer Size", "Buffer Size (samples)", current_buffer_size, self.set_buffer_size, self)
        dialog.exec_()

    def show_about_dtmf_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("About rDTMF")
        
        dialog.setWindowFlags(Qt.Dialog | Qt.WindowCloseButtonHint)
        dialog.setFixedSize(530, 180)

        dialog.setStyleSheet("background-color: #051014; color: #FB8B24;")
        layout = QVBoxLayout(dialog)
        label = QLabel("rDTMF version - 1.2\n---------------------------------------------------------------------------------------------------------------------------------\nDual-Tone Multi-Frequency (DTMF) decoder software is a specialized application designed to\nidentify and interpret DTMF signals, which are commonly used in telephone systems for\ntransmitting digits and control signals. These signals are generated by pressing keys on a\ntelephone keypad, where each key emits a unique combination of two specific frequencies.\n\nThe primary purpose of DTMF decoder software is to analyze these frequencies and translate\nthem into readable digits or commands, enhancing the interaction between users and electronic systems.")
        label.setStyleSheet("color: #FB8B24;")
        layout.addWidget(label)
        ok_button = QDialogButton("Close")
        ok_button.setStyleSheet("background-color: #555287; color: #FB8B24;")
        ok_button.clicked.connect(dialog.accept)
        layout.addWidget(ok_button)
        dialog.exec_()

    def open_author_link(self):
        # Open a URL in the default web browser
        import webbrowser
        webbrowser.open("https://realratnadwip.wordpress.com/")

    def set_sample_rate(self, sample_rate):
        if self.audio_thread:
            self.audio_thread.set_sample_rate(sample_rate)

    def set_timing(self, delay):
        if self.audio_thread:
            self.audio_thread.set_delay(delay)

    def set_buffer_size(self, buffer_size):
        if self.audio_thread:
            self.audio_thread.set_buffer_size(buffer_size)

    def closeEvent(self, event):
        # Ensure the audio thread is stopped before closing
        if self.audio_thread:
            self.audio_thread.stop()
            self.audio_thread.wait()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setWindowIcon(QtGui.QIcon(resource_path('asset\\windico.png')))
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
