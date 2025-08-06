import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
import pandas as pd
import threading
import time
from datetime import datetime
import os
import sys
from PIL import Image, ImageTk
import dlib
from scipy.spatial import distance
import math
import random

# Try to import FER for emotion detection
try:
    from fer import FER
    FER_AVAILABLE = True
except ImportError:
    FER_AVAILABLE = False
    print("FER library not available. Install with: pip install fer")

class MentalFatigueDetector:
    def __init__(self, root):
        self.root = root
        self.root.title("Mental Fatigue Detection System")
        self.root.geometry("1200x800")
        
        # Initialize variables
        self.cap = None
        self.is_recording = False
        self.video_thread = None
        self.current_frame = None
        self.fatigue_level = 0.0
        self.data_log = []
        self.start_time = None
        
        # Initialize face detection and landmark predictor
        self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Try to initialize dlib predictor
        self.predictor = None
        try:
            # You'll need to download this file from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
            self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
            self.dlib_detector = dlib.get_frontal_face_detector()
        except:
            print("dlib predictor not found. Download 'shape_predictor_68_face_landmarks.dat' for advanced features")
        
        # Initialize emotion detector
        if FER_AVAILABLE:
            self.emotion_detector = FER(mtcnn=True)
        else:
            self.emotion_detector = None
        
        # Blink detection variables
        self.blink_counter = 0
        self.blink_total = 0
        self.frame_counter = 0
        self.blink_consecutive_frames = 0
        
        # Eye aspect ratio constants
        self.EAR_THRESHOLD = 0.25
        self.EAR_CONSECUTIVE_FRAMES = 2
        
        self.setup_gui()
        
    def setup_gui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Initialize ttk.Style for custom progress bar colors
        self.style = ttk.Style()
        self._configure_styles()

        # Dictionaries to hold references to Tkinter widgets and variables
        self.sensor_labels = {}
        self.progress_vars = {}
        self.progress_bars = {}
        self.sensor_units = { # Mapping of sensor keys to their units
            "room_temp": " °C",
            "co2_saturation": " ppm",
            "o2neg_saturation": " ions/cm³",
            "humidity_percentage": " %",
        }
        # Store initial base values for simulation fluctuation
        self.base_sensor_values = {
            "room_temp": 22.0,
            "co2_saturation": 700.0,
            "o2neg_saturation": 1200.0,
            "humidity_percentage": 50.0,
        }
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Video source selection
        ttk.Label(control_frame, text="Video Source:").grid(row=0, column=0, sticky=tk.W)
        self.source_var = tk.StringVar(value="webcam")
        source_frame = ttk.Frame(control_frame)
        source_frame.grid(row=0, column=1, sticky=(tk.W, tk.E))
        
        ttk.Radiobutton(source_frame, text="Webcam", variable=self.source_var, value="webcam").pack(side=tk.LEFT)
        ttk.Radiobutton(source_frame, text="Video File", variable=self.source_var, value="file").pack(side=tk.LEFT)
        
        # File selection
        self.file_path = tk.StringVar()
        ttk.Button(control_frame, text="Select Video File", command=self.select_video_file).grid(row=0, column=2, padx=(10, 0))
        
        # Camera index selection
        ttk.Label(control_frame, text="Camera Index:").grid(row=1, column=0, sticky=tk.W)
        self.camera_index = tk.IntVar(value=0)
        ttk.Spinbox(control_frame, from_=0, to=5, textvariable=self.camera_index, width=10).grid(row=1, column=1, sticky=tk.W)
        
        # Control buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=2, column=0, columnspan=3, pady=(10, 0))
        
        self.start_button = ttk.Button(button_frame, text="Start Detection", command=self.start_detection)
        self.start_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_button = ttk.Button(button_frame, text="Stop Detection", command=self.stop_detection, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(button_frame, text="Save CSV", command=self.save_csv).pack(side=tk.LEFT)
        
        # Video display and info panel
        video_info_frame = ttk.Frame(main_frame)
        video_info_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        video_info_frame.columnconfigure(0, weight=2)
        video_info_frame.columnconfigure(1, weight=1)
        video_info_frame.rowconfigure(0, weight=1)
        
        # Video display
        video_frame = ttk.LabelFrame(video_info_frame, text="Video Feed", padding="5")
        video_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        video_frame.rowconfigure(0, weight=1)
        video_frame.columnconfigure(0, weight=1)
        
        self.video_label = ttk.Label(video_frame, text="No video feed")
        self.video_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Info panel
        info_frame = ttk.LabelFrame(video_info_frame, text="Analysis Results", padding="10")
        info_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Mental Fatigue Indicator
        ttk.Label(info_frame, text="Mental Fatigue Level:", font=("Arial", 12, "bold")).pack(anchor=tk.W)
        
        # Progress bar for fatigue level
        self.fatigue_var = tk.DoubleVar()
        self.fatigue_progress = ttk.Progressbar(info_frame, variable=self.fatigue_var, maximum=100, length=200)
        self.fatigue_progress.pack(fill=tk.X, pady=(5, 10))
        
        self.fatigue_label = ttk.Label(info_frame, text="0%", font=("Arial", 11))
        self.fatigue_label.pack(anchor=tk.W)
        
        # Statistics
        stats_frame = ttk.LabelFrame(info_frame, text="Real-time Statistics", padding="5")
        stats_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.stats_labels = {}
        stats_items = [
            ("Faces Detected", "faces_count"),
            ("Avg Eye Openness", "eye_openness"),
            ("Blink Rate (per min)", "blink_rate"),  
            ("Glowing Objects", "glowing_objects"),
            ("Dominant Emotion", "emotion"),
            ("Head Pose", "head_pose")
        ]
        
        for i, (label, key) in enumerate(stats_items):
            ttk.Label(stats_frame, text=f"{label}:").grid(row=i, column=0, sticky=tk.W, pady=2)
            self.stats_labels[key] = ttk.Label(stats_frame, text="--", foreground="blue")
            self.stats_labels[key].grid(row=i, column=1, sticky=tk.W, padx=(10, 0), pady=2)

        # LabelFrame specifically for sensor data display
        sensor_frame = ttk.LabelFrame(info_frame, text="Sensor Data", padding="5")
        sensor_frame.pack(fill=tk.X, pady=(10, 0))

         # Define sensor items with their display labels, data keys, and maximum values for progress bars
        sensor_items = [
            ("Temperature", "room_temp", 30), # Max value for temperature (e.g., up to 30°C)
            ("CO₂", "co2_saturation", 2000), # Max value for CO2 (e.g., up to 2000 ppm)
            ("O₂⁻", "o2neg_saturation", 2000), # Max value for O2- ions (e.g., up to 2000 ions/cm³)
            ("Humidity", "humidity_percentage", 100), # Max value for humidity (0-100%)
        ]

        # Loop through each sensor item to create its label, value display, and progress bar
        for i, (label_text, key, max_value) in enumerate(sensor_items):
            # Sensor Name Label (e.g., "Temperature:")
            ttk.Label(sensor_frame, text=f"{label_text}:").grid(row=i, column=0, sticky=tk.W, pady=2, padx=5)

            # Current Value Label (e.g., "22.5")
            self.sensor_labels[key] = ttk.Label(sensor_frame, text="--", foreground="blue")
            self.sensor_labels[key].grid(row=i, column=1, sticky=tk.W, padx=(10, 0), pady=2)

            # Progress Bar for the sensor
            self.progress_vars[key] = tk.DoubleVar() # Variable to control progress bar value
            progress_bar = ttk.Progressbar(sensor_frame, variable=self.progress_vars[key], maximum=max_value, length=200)
            # Use grid for the progress bar to be consistent with other widgets in sensor_frame
            progress_bar.grid(row=i, column=2, sticky=tk.EW, padx=(10, 5), pady=2)
            self.progress_bars[key] = progress_bar # Store reference to the progress bar

        # Configure the third column (where progress bars are) to expand when the window resizes
        sensor_frame.grid_columnconfigure(2, weight=1)

    def update_sensor_data(self, data):
        """
        Updates the sensor display with new data.
        Iterates through the provided data, updates labels and progress bars,
        and applies the correct color style based on the value.

        Args:
            data (dict): A dictionary where keys are sensor identifiers
                         (e.g., "room_temp") and values are their measurements.
        """
        for key, value in data.items():
            if key in self.progress_vars:
                # Update the text label displaying the current value with its unit
                unit = self.sensor_units.get(key, "") # Get the unit, default to empty string if not found
                self.sensor_labels[key].config(text=f"{value:.1f}{unit}") # Format to one decimal place and add unit

                # Update the progress bar's value
                self.progress_vars[key].set(value)

                # Determine and apply the appropriate style (color) to the progress bar
                self._set_progressbar_style(key, value)

    def _set_progressbar_style(self, key, value):
        """
        Determines the appropriate progress bar style (color) based on the
        sensor type and its current value, applying comfort level logic.

        Args:
            key (str): The identifier for the sensor (e.g., "room_temp").
            value (float): The current measured value of the sensor.
        """
        style_name = "Horizontal.TProgressbar" # Default style if no specific condition met

        if key == "room_temp":
            # Room Temperature: Green (comfortable), Yellow (acceptable), Dark Blue (too cold), Red (too hot)
            if 20 <= value <= 24:
                style_name = "green.Horizontal.TProgressbar"
            elif (18 <= value < 20) or (24 < value <= 26):
                style_name = "yellow.Horizontal.TProgressbar"
            elif value < 18:
                style_name = "darkblue.Horizontal.TProgressbar"
            else: # value > 26
                style_name = "red.Horizontal.TProgressbar"
        elif key == "co2_saturation":
            # CO2 Saturation: Green (low), Yellow (open windows), Red (uncomfortable)
            if value < 800:
                style_name = "green.Horizontal.TProgressbar"
            elif 800 <= value <= 1200:
                style_name = "yellow.Horizontal.TProgressbar"
            else: # value > 1200
                style_name = "red.Horizontal.TProgressbar"
        elif key == "o2neg_saturation":
            # Negative O2 Ions: Green (beneficial), Yellow (acceptable), Red (low)
            # Research suggests >1000 ions/cm³ is beneficial, 500-1000 acceptable, <500 low.
            if value > 1000:
                style_name = "green.Horizontal.TProgressbar"
            elif 500 <= value <= 1000:
                style_name = "yellow.Horizontal.TProgressbar"
            else: # value < 500
                style_name = "red.Horizontal.TProgressbar"
        elif key == "humidity_percentage":
            # Humidity: Green (convenient), Yellow (uncomfortable), Red (too dry/too high)
            if 40 <= value <= 60:
                style_name = "green.Horizontal.TProgressbar"
            elif (30 <= value < 40) or (60 < value <= 70):
                style_name = "yellow.Horizontal.TProgressbar"
            else: # value < 30 or value > 70
                style_name = "red.Horizontal.TProgressbar"

        # Apply the determined style to the specific progress bar
        self.progress_bars[key].config(style=style_name)

    def _configure_styles(self):
        """
        Configures custom styles for ttk.Progressbar to allow dynamic coloring.
        Each style defines the background color of the progress bar itself.
        """
        # Green style for comfortable conditions
        self.style.configure("green.Horizontal.TProgressbar", troughcolor='lightgray', background='green')
        # Yellow style for acceptable/warning conditions
        self.style.configure("yellow.Horizontal.TProgressbar", troughcolor='lightgray', background='yellow')
        # Red style for uncomfortable/critical conditions
        self.style.configure("red.Horizontal.TProgressbar", troughcolor='lightgray', background='red')
        # Dark blue style for too cold temperature
        self.style.configure("darkblue.Horizontal.TProgressbar", troughcolor='lightgray', background='darkblue')

    
    def select_video_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"), ("All files", "*.*")]
        )
        if file_path:
            self.file_path.set(file_path)
    
    def calculate_ear(self, eye_points):
        """Calculate Eye Aspect Ratio"""
        # Vertical eye landmarks
        A = distance.euclidean(eye_points[1], eye_points[5])
        B = distance.euclidean(eye_points[2], eye_points[4])
        # Horizontal eye landmark
        C = distance.euclidean(eye_points[0], eye_points[3])
        # Eye aspect ratio
        ear = (A + B) / (2.0 * C)
        return ear
    
    def detect_blinks(self, landmarks):
        """Detect blinks using eye aspect ratio"""
        if landmarks is None:
            return 0, 0
        
        # Eye landmark indices (68-point model)
        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]
        
        # Calculate EAR for both eyes
        left_ear = self.calculate_ear(left_eye)
        right_ear = self.calculate_ear(right_eye)
        ear = (left_ear + right_ear) / 2.0
        
        # Check for blink
        if ear < self.EAR_THRESHOLD:
            self.blink_consecutive_frames += 1
        else:
            if self.blink_consecutive_frames >= self.EAR_CONSECUTIVE_FRAMES:
                self.blink_total += 1
            self.blink_consecutive_frames = 0
        
        return ear, self.blink_total
    
    def detect_glowing_objects(self, frame):
        """Detect bright/glowing objects like screens"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (11, 11), 0)
        
        # Threshold for bright areas
        _, bright_areas = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(bright_areas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area (to avoid noise)
        glowing_objects = [c for c in contours if cv2.contourArea(c) > 500]
        
        return len(glowing_objects), glowing_objects
    
    def calculate_head_pose(self, landmarks):
        """Calculate head pose estimation"""
        if landmarks is None or len(landmarks) < 68:
            return "Unknown"
        
        # 3D model points
        model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ])
        
        # 2D image points from landmarks
        image_points = np.array([
            landmarks[30],     # Nose tip
            landmarks[8],      # Chin
            landmarks[36],     # Left eye left corner
            landmarks[45],     # Right eye right corner
            landmarks[48],     # Left mouth corner
            landmarks[54]      # Right mouth corner
        ], dtype="double")
        
        # Camera matrix (approximate)
        size = (640, 480)  # Assume standard resolution
        focal_length = size[1]
        center = (size[1]/2, size[0]/2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")
        
        dist_coeffs = np.zeros((4,1))
        
        try:
            success, rotation_vector, translation_vector = cv2.solvePnP(
                model_points, image_points, camera_matrix, dist_coeffs)
            
            if success:
                # Convert rotation vector to rotation matrix
                rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
                
                # Calculate Euler angles
                angles = cv2.RQDecomp3x3(rotation_matrix)[0]
                
                # Determine head pose
                if angles[1] > 10:
                    return "Looking Right"
                elif angles[1] < -10:
                    return "Looking Left"
                elif angles[0] > 10:
                    return "Looking Up"
                elif angles[0] < -10:
                    return "Looking Down"
                else:
                    return "Forward"
        except:
            pass
        
        return "Unknown"
    
    def calculate_fatigue_level(self, emotion_data, eye_openness, blink_rate, head_pose):
        """Calculate mental fatigue level based on multiple factors"""
        fatigue_score = 0.0
        
        # Emotion-based fatigue (30% weight)
        if emotion_data:
            # Higher fatigue for tired-looking emotions
            emotion_weights = {
                'sad': 0.7,
                'neutral': 0.4,
                'angry': 0.6,
                'fear': 0.5,
                'surprise': 0.2,
                'disgust': 0.3,
                'happy': 0.1
            }
            
            for emotion, confidence in emotion_data.items():
                if emotion in emotion_weights:
                    fatigue_score += confidence * emotion_weights[emotion] * 0.3
        
        # Eye openness (25% weight)
        if eye_openness < 0.2:  # Very closed eyes
            fatigue_score += 0.25
        elif eye_openness < 0.25:  # Partially closed
            fatigue_score += 0.15
        
        # Abnormal blink rate (25% weight)
        normal_blink_rate = 17  # Average blinks per minute
        if blink_rate > normal_blink_rate * 1.5:  # Too many blinks
            fatigue_score += 0.2
        elif blink_rate < normal_blink_rate * 0.5:  # Too few blinks
            fatigue_score += 0.15
        
        # Head pose (20% weight)
        if head_pose == "Looking Down":
            fatigue_score += 0.2
        elif head_pose in ["Looking Right", "Looking Left"]:
            fatigue_score += 0.1
        
        return min(fatigue_score * 100, 100)  # Convert to percentage, cap at 100%
    
    def process_frame(self, frame):
        """Process a single frame for fatigue detection"""
        if frame is None:
            return frame
        
        # Convert to RGB for display
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(gray, 1.1, 4)
        
        face_count = len(faces)
        avg_eye_openness = 0
        emotion_data = {}
        head_pose = "Unknown"
        
        # Process each face
        for (x, y, w, h) in faces:
            # Draw rectangle around face
            cv2.rectangle(rgb_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Emotion detection
            if self.emotion_detector:
                try:
                    face_region = frame[y:y+h, x:x+w]
                    emotions = self.emotion_detector.detect_emotions(face_region)
                    if emotions:
                        emotion_data = emotions[0]['emotions']
                        dominant_emotion = max(emotion_data, key=emotion_data.get)
                        confidence = emotion_data[dominant_emotion]
                        
                        # Display emotion
                        cv2.putText(rgb_frame, f"{dominant_emotion}: {confidence:.2f}", 
                                  (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                except:
                    pass
            
            # Landmark detection for advanced features
            if self.predictor:
                try:
                    dlib_rect = dlib.rectangle(x, y, x + w, y + h)
                    landmarks = self.predictor(gray, dlib_rect)
                    
                    # Convert landmarks to numpy array
                    landmark_points = []
                    for n in range(68):
                        point = landmarks.part(n)
                        landmark_points.append([point.x, point.y])
                    landmark_points = np.array(landmark_points)
                    
                    # Eye openness and blink detection
                    eye_openness, total_blinks = self.detect_blinks(landmark_points)
                    avg_eye_openness = eye_openness
                    
                    # Head pose estimation
                    head_pose = self.calculate_head_pose(landmark_points)
                    
                    # Draw eye landmarks
                    for point in landmark_points[36:48]:  # Eye landmarks
                        cv2.circle(rgb_frame, tuple(point), 1, (0, 255, 255), -1)
                        
                except:
                    pass
        
        # Detect glowing objects
        glowing_count, glowing_contours = self.detect_glowing_objects(frame)
        
        # Draw glowing objects
        for contour in glowing_contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(rgb_frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(rgb_frame, "Glowing", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Calculate blink rate (per minute)
        if self.start_time:
            elapsed_minutes = (time.time() - self.start_time) / 60.0
            blink_rate = self.blink_total / max(elapsed_minutes, 1/60)  # Avoid division by zero
        else:
            blink_rate = 0
        
        # Calculate fatigue level
        self.fatigue_level = self.calculate_fatigue_level(emotion_data, avg_eye_openness, blink_rate, head_pose)
        
        # Update GUI
        self.update_stats(face_count, avg_eye_openness, blink_rate, glowing_count, emotion_data, head_pose)
        
        # Log data
        if self.is_recording:
            current_time = time.time() - self.start_time if self.start_time else 0
            log_entry = {
                'Recording Time': current_time,
                'Amount of Recorded Faces': face_count,
                'Average Eye Openness': avg_eye_openness,
                'Average Mental Fatigue Level': self.fatigue_level,
                'Amount of Glowing Objects': glowing_count,
                'Average Total Blinking Rate': blink_rate,
                'Dominant Emotion': max(emotion_data, key=emotion_data.get) if emotion_data else 'Unknown',
                'Head Pose': head_pose,
                'Frame Number': self.frame_counter
            }
            self.data_log.append(log_entry)
        
        self.frame_counter += 1
        return rgb_frame
    
    def update_stats(self, face_count, eye_openness, blink_rate, glowing_count, emotion_data, head_pose):
        """Update GUI statistics"""
        self.stats_labels['faces_count'].config(text=str(face_count))
        self.stats_labels['eye_openness'].config(text=f"{eye_openness:.3f}" if eye_openness > 0 else "--")
        self.stats_labels['blink_rate'].config(text=f"{blink_rate:.1f}")
        self.stats_labels['glowing_objects'].config(text=str(glowing_count))
        
        if emotion_data:
            dominant_emotion = max(emotion_data, key=emotion_data.get)
            confidence = emotion_data[dominant_emotion]
            self.stats_labels['emotion'].config(text=f"{dominant_emotion} ({confidence:.2f})")
        else:
            self.stats_labels['emotion'].config(text="--")
        
        self.stats_labels['head_pose'].config(text=head_pose)
        
        # Update fatigue level
        self.fatigue_var.set(self.fatigue_level)
        self.fatigue_label.config(text=f"{self.fatigue_level:.1f}%")
        
        # Color code fatigue level
        if self.fatigue_level < 30:
            color = "green"
        elif self.fatigue_level < 60:
            color = "orange"  
        else:
            color = "red"
        self.fatigue_label.config(foreground=color)
    
    def video_loop(self):
        """Main video processing loop"""
        while self.is_recording and self.cap:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Process frame
            processed_frame = self.process_frame(frame)
            
            if processed_frame is not None:
                # Resize for display
                height, width = processed_frame.shape[:2]
                display_width = 600
                display_height = int(height * display_width / width)
                processed_frame = cv2.resize(processed_frame, (display_width, display_height))
                
                # Convert to PhotoImage for tkinter
                self.current_frame = processed_frame
                image = Image.fromarray(processed_frame)
                photo = ImageTk.PhotoImage(image)
                
                # Update display in main thread
                self.root.after(0, self.update_video_display, photo)
            
            # Control frame rate
            time.sleep(0.03)  # ~30 FPS
    
    def update_video_display(self, photo):
        """Update video display (called from main thread)"""
        self.video_label.configure(image=photo)
        self.video_label.image = photo  # Keep a reference
    
    def start_detection(self):
        """Start video capture and processing"""
        try:
            # Initialize video capture
            if self.source_var.get() == "webcam":
                self.cap = cv2.VideoCapture(self.camera_index.get())
            else:
                if not self.file_path.get():
                    messagebox.showerror("Error", "Please select a video file")
                    return
                self.cap = cv2.VideoCapture(self.file_path.get())
            
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not open video source")
                return
            
            # Reset counters
            self.blink_total = 0
            self.frame_counter = 0
            self.data_log = []
            self.start_time = time.time()
            
            # Start recording
            self.is_recording = True
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            
            # Start video processing thread
            self.video_thread = threading.Thread(target=self.video_loop)
            self.video_thread.daemon = True
            self.video_thread.start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start detection: {str(e)}")
    
    def stop_detection(self):
        """Stop video capture and processing"""
        self.is_recording = False
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        if self.video_thread:
            self.video_thread.join(timeout=1.0)
        
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        
        # Clear video display
        self.video_label.config(image="", text="Detection stopped")
        
        messagebox.showinfo("Detection Stopped", f"Recorded {len(self.data_log)} frames of data")
    
    def save_csv(self):
        """Save logged data to CSV file"""
        if not self.data_log:
            messagebox.showwarning("No Data", "No data to save")
            return
        
        try:
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"mental_fatigue_data_{timestamp}.csv"
            
            # Convert to DataFrame and save
            df = pd.DataFrame(self.data_log)
            df.to_csv(filename, index=False)
            
            messagebox.showinfo("Data Saved", f"Data saved to {filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save data: {str(e)}")

def main():
    # Check dependencies
    missing_deps = []
    
    if not FER_AVAILABLE:
        missing_deps.append("fer (pip install fer)")
    
    try:
        import dlib
    except ImportError:
        missing_deps.append("dlib (pip install dlib)")
    
    if missing_deps:
        print("Warning: Missing optional dependencies:")
        for dep in missing_deps:
            print(f"  - {dep}")
        print("Some features may not work properly.")
        print()
    
    # Create and run application
    root = tk.Tk()
    app = MentalFatigueDetector(root)

    # Function to simulate sensor data updates periodically
    def simulate_data():
        """
        Generates random sensor data within a narrow fluctuation range
        and updates the dashboard.
        """
        new_data = {}
        fluctuation_percentage = 0.05 # +/- 5% fluctuation

        for key, base_value in app.base_sensor_values.items():
            # Calculate min and max for fluctuation
            min_val = base_value * (1 - fluctuation_percentage)
            max_val = base_value * (1 + fluctuation_percentage)

            # Ensure values stay within reasonable overall bounds
            if key == "room_temp":
                min_val = max(min_val, 15)
                max_val = min(max_val, 30)
            elif key == "co2_saturation":
                min_val = max(min_val, 400)
                max_val = min(max_val, 2000)
            elif key == "o2neg_saturation":
                min_val = max(min_val, 200)
                max_val = min(max_val, 2000)
            elif key == "humidity_percentage":
                min_val = max(min_val, 0)
                max_val = min(max_val, 100)

            # Generate new value within the fluctuating range
            new_data[key] = random.uniform(min_val, max_val)

        app.update_sensor_data(new_data)
        # Schedule the next data simulation after 2000 milliseconds (2 seconds)
        root.after(2000, simulate_data)

    # Start the data simulation
    simulate_data()

    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("Application interrupted")
    finally:
        if app.cap:
            app.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
