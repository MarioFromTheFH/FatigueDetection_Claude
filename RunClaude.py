#!/usr/bin/env python3
"""
Mental Fatigue Detection GUI
A cross-platform application for real-time emotion detection and mental fatigue estimation.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
from PIL import Image, ImageTk
import numpy as np
import threading
import time
from collections import deque
import sys
import os

try:
    from fer import FER
except ImportError:
    print("FER library not installed. Install with: pip install fer")
    sys.exit(1)

class MentalFatigueDetector:
    def __init__(self, root):
        self.root = root
        self.root.title("Mental Fatigue Detection System")
        self.root.geometry("900x700")
        self.root.configure(bg='#2c3e50')
        
        # Video capture variables
        self.cap = None
        self.is_running = False
        self.current_frame = None
        self.video_source = 0  # Default webcam
        
        # Emotion detection
        self.detector = FER(mtcnn=True)  # Use MTCNN for better face detection
        
        # Mental fatigue calculation
        self.emotion_history = deque(maxlen=30)  # Store last 30 emotion readings
        self.fatigue_score = 0.0
        
        # Threading
        self.video_thread = None
        self.processing_thread = None
        
        self.setup_gui()
        
    def setup_gui(self):
        """Setup the main GUI interface"""
        # Main title
        title_label = tk.Label(
            self.root, 
            text="Mental Fatigue Detection System", 
            font=("Arial", 20, "bold"),
            bg='#2c3e50',
            fg='white'
        )
        title_label.pack(pady=10)
        
        # Control frame
        control_frame = tk.Frame(self.root, bg='#2c3e50')
        control_frame.pack(pady=10)
        
        # Video source selection
        source_frame = tk.Frame(control_frame, bg='#2c3e50')
        source_frame.pack(side=tk.LEFT, padx=20)
        
        tk.Label(
            source_frame, 
            text="Video Source:", 
            font=("Arial", 12),
            bg='#2c3e50',
            fg='white'
        ).pack()
        
        self.source_var = tk.StringVar(value="Webcam")
        source_combo = ttk.Combobox(
            source_frame, 
            textvariable=self.source_var,
            values=["Webcam", "Video File"],
            state="readonly",
            width=15
        )
        source_combo.pack(pady=5)
        source_combo.bind('<<ComboboxSelected>>', self.on_source_change)
        
        # Control buttons
        button_frame = tk.Frame(control_frame, bg='#2c3e50')
        button_frame.pack(side=tk.LEFT, padx=20)
        
        self.start_button = tk.Button(
            button_frame,
            text="Start Detection",
            command=self.start_detection,
            bg='#27ae60',
            fg='white',
            font=("Arial", 12, "bold"),
            padx=20,
            pady=5
        )
        self.start_button.pack(pady=2)
        
        self.stop_button = tk.Button(
            button_frame,
            text="Stop Detection",
            command=self.stop_detection,
            bg='#e74c3c',
            fg='white',
            font=("Arial", 12, "bold"),
            padx=20,
            pady=5,
            state=tk.DISABLED
        )
        self.stop_button.pack(pady=2)
        
        # Video display frame
        video_frame = tk.Frame(self.root, bg='#34495e', relief=tk.RAISED, bd=2)
        video_frame.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)
        
        self.video_label = tk.Label(
            video_frame,
            text="No video feed",
            bg='#34495e',
            fg='white',
            font=("Arial", 16)
        )
        self.video_label.pack(expand=True)
        
        # Mental fatigue indicator frame
        fatigue_frame = tk.Frame(self.root, bg='#2c3e50')
        fatigue_frame.pack(pady=10, padx=20, fill=tk.X)
        
        # Fatigue level label
        self.fatigue_label = tk.Label(
            fatigue_frame,
            text="Mental Fatigue Level: 0%",
            font=("Arial", 16, "bold"),
            bg='#2c3e50',
            fg='white'
        )
        self.fatigue_label.pack()
        
        # Fatigue progress bar
        self.fatigue_progress = ttk.Progressbar(
            fatigue_frame,
            length=400,
            mode='determinate',
            style='Fatigue.Horizontal.TProgressbar'
        )
        self.fatigue_progress.pack(pady=10)
        
        # Configure progress bar colors
        self.setup_progress_bar_style()
        
        # Current emotions display
        emotions_frame = tk.Frame(self.root, bg='#2c3e50')
        emotions_frame.pack(pady=10)
        
        tk.Label(
            emotions_frame,
            text="Current Emotions:",
            font=("Arial", 12, "bold"),
            bg='#2c3e50',
            fg='white'
        ).pack()
        
        self.emotions_text = tk.Text(
            emotions_frame,
            height=4,
            width=60,
            bg='#34495e',
            fg='white',
            font=("Arial", 10)
        )
        self.emotions_text.pack(pady=5)
        
    def setup_progress_bar_style(self):
        """Setup custom style for progress bar"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure progress bar to change color based on value
        style.configure(
            'Fatigue.Horizontal.TProgressbar',
            background='#27ae60',  # Green for low fatigue
            troughcolor='#34495e',
            borderwidth=0,
            lightcolor='#27ae60',
            darkcolor='#27ae60'
        )
        
    def update_progress_bar_color(self, value):
        """Update progress bar color based on fatigue level"""
        style = ttk.Style()
        
        if value < 30:
            color = '#27ae60'  # Green
        elif value < 70:
            color = '#f39c12'  # Orange
        else:
            color = '#e74c3c'  # Red
            
        style.configure(
            'Fatigue.Horizontal.TProgressbar',
            background=color,
            lightcolor=color,
            darkcolor=color
        )
    
    def on_source_change(self, event=None):
        """Handle video source change"""
        if self.is_running:
            messagebox.showwarning("Warning", "Stop detection before changing source")
            return
            
        if self.source_var.get() == "Video File":
            file_path = filedialog.askopenfilename(
                title="Select Video File",
                filetypes=[
                    ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                    ("All files", "*.*")
                ]
            )
            if file_path:
                self.video_source = file_path
            else:
                self.source_var.set("Webcam")
                self.video_source = 0
        else:
            self.video_source = 0
    
    def calculate_mental_fatigue(self, emotions):
        """
        Calculate mental fatigue based on emotion scores
        Combines boredom (neutral + sad) and tiredness indicators
        """
        if not emotions:
            return 0.0
        
        # Extract emotion scores
        neutral = emotions.get('neutral', 0)
        sad = emotions.get('sad', 0)
        angry = emotions.get('angry', 0)
        fear = emotions.get('fear', 0)
        happy = emotions.get('happy', 0)
        
        # Calculate fatigue components
        # Boredom: high neutral, low happy
        boredom_score = neutral * (1 - happy)
        
        # Tiredness: combination of sad, angry, and fear (stress indicators)
        tiredness_score = (sad + angry + fear) / 3
        
        # Combined mental fatigue (weighted average)
        mental_fatigue = (boredom_score * 0.6 + tiredness_score * 0.4)
        
        # Normalize to 0-100 scale
        return min(mental_fatigue * 100, 100)
    
    def process_frame(self, frame):
        """Process frame for emotion detection"""
        try:
            # Detect emotions
            result = self.detector.detect_emotions(frame)
            
            if result:
                # Get the face with highest confidence
                face_data = max(result, key=lambda x: max(x['emotions'].values()))
                emotions = face_data['emotions']
                box = face_data['box']
                
                # Calculate mental fatigue
                fatigue = self.calculate_mental_fatigue(emotions)
                self.emotion_history.append(fatigue)
                
                # Smooth fatigue score using moving average
                if len(self.emotion_history) > 5:
                    self.fatigue_score = np.mean(list(self.emotion_history)[-10:])
                else:
                    self.fatigue_score = fatigue
                
                # Draw bounding box on frame
                x, y, w, h = box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Add emotion text
                emotion_text = f"Fatigue: {self.fatigue_score:.1f}%"
                cv2.putText(frame, emotion_text, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Update GUI in main thread
                self.root.after(0, self.update_gui, emotions)
                
        except Exception as e:
            print(f"Error processing frame: {e}")
        
        return frame
    
    def update_gui(self, emotions):
        """Update GUI elements with current emotion data"""
        # Update fatigue level
        fatigue_percent = int(self.fatigue_score)
        self.fatigue_label.config(text=f"Mental Fatigue Level: {fatigue_percent}%")
        self.fatigue_progress['value'] = fatigue_percent
        self.update_progress_bar_color(fatigue_percent)
        
        # Update emotions display
        self.emotions_text.delete(1.0, tk.END)
        emotions_str = "Detected Emotions:\n"
        for emotion, score in emotions.items():
            emotions_str += f"{emotion.capitalize()}: {score:.3f}\n"
        emotions_str += f"\nCalculated Mental Fatigue: {self.fatigue_score:.1f}%"
        self.emotions_text.insert(tk.END, emotions_str)
    
    def video_capture_loop(self):
        """Main video capture and processing loop"""
        while self.is_running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    if isinstance(self.video_source, str):  # Video file ended
                        break
                    continue
                
                # Flip frame for webcam (mirror effect)
                if self.video_source == 0:
                    frame = cv2.flip(frame, 1)
                
                # Process frame for emotion detection
                processed_frame = self.process_frame(frame.copy())
                
                # Convert frame for tkinter display
                frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                
                # Resize frame to fit display
                display_size = (640, 480)
                frame_pil = frame_pil.resize(display_size, Image.Resampling.LANCZOS)
                
                frame_tk = ImageTk.PhotoImage(frame_pil)
                
                # Update video display in main thread
                self.root.after(0, self.update_video_display, frame_tk)
                
                # Control frame rate
                time.sleep(0.03)  # ~30 FPS
                
            except Exception as e:
                print(f"Error in video loop: {e}")
                break
        
        # Cleanup
        if self.cap:
            self.cap.release()
        self.root.after(0, self.video_stopped)
    
    def update_video_display(self, frame_tk):
        """Update video display widget"""
        self.video_label.configure(image=frame_tk, text="")
        self.video_label.image = frame_tk  # Keep a reference
    
    def start_detection(self):
        """Start video capture and emotion detection"""
        try:
            # Initialize video capture
            self.cap = cv2.VideoCapture(self.video_source)
            
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not open video source")
                return
            
            # Set video properties for better performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Update UI
            self.is_running = True
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            
            # Start video capture thread
            self.video_thread = threading.Thread(target=self.video_capture_loop)
            self.video_thread.daemon = True
            self.video_thread.start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start detection: {str(e)}")
    
    def stop_detection(self):
        """Stop video capture and emotion detection"""
        self.is_running = False
        
        # Wait for thread to finish
        if self.video_thread and self.video_thread.is_alive():
            self.video_thread.join(timeout=2)
    
    def video_stopped(self):
        """Called when video capture stops"""
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.video_label.configure(image="", text="No video feed")
        self.video_label.image = None
    
    def on_closing(self):
        """Handle application closing"""
        self.stop_detection()
        self.root.destroy()

def main():
    """Main application entry point"""
    # Check for required libraries
    required_libs = ['cv2', 'PIL', 'fer', 'numpy']
    missing_libs = []
    
    for lib in required_libs:
        try:
            __import__(lib)
        except ImportError:
            missing_libs.append(lib)
    
    if missing_libs:
        print("Missing required libraries:")
        for lib in missing_libs:
            if lib == 'cv2':
                print("  Install OpenCV: pip install opencv-python")
            elif lib == 'PIL':
                print("  Install Pillow: pip install Pillow")
            elif lib == 'fer':
                print("  Install FER: pip install fer")
            elif lib == 'numpy':
                print("  Install NumPy: pip install numpy")
        sys.exit(1)
    
    # Create and run application
    root = tk.Tk()
    app = MentalFatigueDetector(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
