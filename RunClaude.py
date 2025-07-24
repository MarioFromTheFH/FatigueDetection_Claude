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

try:
    import dlib
    import face_recognition
    EYE_DETECTION_AVAILABLE = True
except ImportError:
    print("Eye detection libraries not available. Install with: pip install dlib face-recognition")
    print("Eye opening detection will be disabled.")
    EYE_DETECTION_AVAILABLE = False

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
        self.eye_openness_history = deque(maxlen=15)  # Store eye openness readings
        self.detected_faces_count = 0
        
        # Eye detection setup
        if EYE_DETECTION_AVAILABLE:
            self.face_landmarks_predictor = dlib.shape_predictor(self.get_landmarks_model())
        else:
            self.face_landmarks_predictor = None
        
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
            text="Detection Results:",
            font=("Arial", 12, "bold"),
            bg='#2c3e50',
            fg='white'
        ).pack()
        
        self.emotions_text = tk.Text(
            emotions_frame,
            height=6,
            width=70,
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
                    ("Video files", "*.mp4 *.MP4 *.avi *.AVI *.mov *.MOV *.mkv *.MKV *.wmv *.WMV"),
                    ("All files", "*.*")
                ]
            )
            if file_path:
                self.video_source = file_path
            else:
                self.source_var.set("Webcam")
                self.video_source = 0
    
    def get_landmarks_model(self):
        """Download and return path to facial landmarks model"""
        import urllib.request
        import os
        
        model_path = "shape_predictor_68_face_landmarks.dat"
        
        if not os.path.exists(model_path):
            print("Downloading facial landmarks model (this may take a while)...")
            url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
            try:
                urllib.request.urlretrieve(url, "shape_predictor_68_face_landmarks.dat.bz2")
                
                # Extract bz2 file
                import bz2
                with bz2.BZ2File("shape_predictor_68_face_landmarks.dat.bz2", 'rb') as f:
                    with open(model_path, 'wb') as out_file:
                        out_file.write(f.read())
                
                # Clean up
                os.remove("shape_predictor_68_face_landmarks.dat.bz2")
                print("Model downloaded successfully!")
                
            except Exception as e:
                print(f"Failed to download landmarks model: {e}")
                print("Eye detection will be disabled.")
                return None
        
        return model_path
    
    def calculate_eye_aspect_ratio(self, eye_points):
        """Calculate Eye Aspect Ratio (EAR) to determine eye openness"""
        # Compute the euclidean distances between the two sets of
        # vertical eye landmarks (x, y)-coordinates
        A = np.linalg.norm(eye_points[1] - eye_points[5])
        B = np.linalg.norm(eye_points[2] - eye_points[4])
        
        # Compute the euclidean distance between the horizontal
        # eye landmark (x, y)-coordinates
        C = np.linalg.norm(eye_points[0] - eye_points[3])
        
        # Compute the eye aspect ratio
        ear = (A + B) / (2.0 * C)
        return ear
    
    def get_eye_openness(self, frame, face_locations):
        """Calculate average eye openness for all detected faces"""
        if not EYE_DETECTION_AVAILABLE or not self.face_landmarks_predictor:
            return None
        
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            eye_openness_scores = []
            
            for face_location in face_locations:
                # Convert face_recognition format to dlib format
                top, right, bottom, left = face_location
                dlib_rect = dlib.rectangle(left, top, right, bottom)
                
                # Get facial landmarks
                landmarks = self.face_landmarks_predictor(gray, dlib_rect)
                
                # Extract left and right eye coordinates
                left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
                right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])
                
                # Calculate eye aspect ratio for both eyes
                left_ear = self.calculate_eye_aspect_ratio(left_eye)
                right_ear = self.calculate_eye_aspect_ratio(right_eye)
                
                # Average the eye aspect ratio together for both eyes
                avg_ear = (left_ear + right_ear) / 2.0
                
                # Convert EAR to eye openness percentage (normalized)
                # EAR typically ranges from ~0.15 (closed) to ~0.35 (wide open)
                eye_openness = min(max((avg_ear - 0.15) / (0.35 - 0.15), 0), 1) * 100
                eye_openness_scores.append(eye_openness)
            
            return np.mean(eye_openness_scores) if eye_openness_scores else None
            
        except Exception as e:
            print(f"Error calculating eye openness: {e}")
            return None
        else:
            self.video_source = 0
    
    def calculate_mental_fatigue(self, all_emotions_data, eye_openness=None):
        """
        Calculate mental fatigue based on emotion scores from all detected faces
        Combines boredom (neutral + sad) and tiredness indicators
        Now includes eye openness as a fatigue indicator
        """
        if not all_emotions_data:
            return 0.0, 0
        
        face_fatigue_scores = []
        
        # Calculate fatigue for each detected face
        for emotions in all_emotions_data:
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
            
            # Combined emotional fatigue (weighted average)
            emotional_fatigue = (boredom_score * 0.6 + tiredness_score * 0.4)
            
            face_fatigue_scores.append(emotional_fatigue)
        
        # Average fatigue across all faces
        avg_emotional_fatigue = np.mean(face_fatigue_scores)
        
        # Incorporate eye openness if available
        if eye_openness is not None:
            # Store eye openness history
            self.eye_openness_history.append(eye_openness)
            
            # Calculate smoothed eye openness
            if len(self.eye_openness_history) > 3:
                smoothed_eye_openness = np.mean(list(self.eye_openness_history)[-5:])
            else:
                smoothed_eye_openness = eye_openness
            
            # Convert eye openness to fatigue indicator
            # Lower eye openness = higher fatigue
            eye_fatigue = (100 - smoothed_eye_openness) / 100
            
            # Combine emotional fatigue with eye fatigue
            # Weight: 70% emotions, 30% eye openness
            combined_fatigue = (avg_emotional_fatigue * 0.7 + eye_fatigue * 0.3)
        else:
            combined_fatigue = avg_emotional_fatigue
        
        # Normalize to 0-100 scale
        fatigue_percentage = min(combined_fatigue * 100, 100)
        
        return fatigue_percentage, len(all_emotions_data)
    
    def process_frame(self, frame):
        """Process frame for emotion detection and eye openness analysis"""
        try:
            # Detect emotions for all faces
            emotion_results = self.detector.detect_emotions(frame)
            
            if emotion_results:
                # Extract all emotions and face locations
                all_emotions = []
                face_locations = []
                
                # Draw bounding boxes and collect data
                for face_data in emotion_results:
                    emotions = face_data['emotions']
                    box = face_data['box']
                    
                    all_emotions.append(emotions)
                    
                    # Convert box format for face_recognition library
                    x, y, w, h = box
                    face_locations.append((y, x + w, y + h, x))  # top, right, bottom, left
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # Add individual face emotion info
                    dominant_emotion = max(emotions, key=emotions.get)
                    confidence = emotions[dominant_emotion]
                    cv2.putText(frame, f"{dominant_emotion}: {confidence:.2f}", 
                               (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Calculate eye openness for all faces
                eye_openness = self.get_eye_openness(frame, face_locations)
                
                # Calculate mental fatigue for all detected faces
                fatigue, face_count = self.calculate_mental_fatigue(all_emotions, eye_openness)
                self.emotion_history.append(fatigue)
                self.detected_faces_count = face_count
                
                # Smooth fatigue score using moving average
                if len(self.emotion_history) > 5:
                    self.fatigue_score = np.mean(list(self.emotion_history)[-10:])
                else:
                    self.fatigue_score = fatigue
                
                # Add overall fatigue text to frame
                fatigue_text = f"Avg Fatigue: {self.fatigue_score:.1f}% ({face_count} faces)"
                cv2.putText(frame, fatigue_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                if eye_openness is not None:
                    eye_text = f"Avg Eye Openness: {eye_openness:.1f}%"
                    cv2.putText(frame, eye_text, (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                # Update GUI in main thread
                self.root.after(0, self.update_gui, all_emotions, eye_openness)
            else:
                # No faces detected
                self.detected_faces_count = 0
                cv2.putText(frame, "No faces detected", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
        except Exception as e:
            print(f"Error processing frame: {e}")
        
        return frame
    
    def update_gui(self, all_emotions, eye_openness=None):
        """Update GUI elements with current emotion data from all faces"""
        # Update fatigue level
        fatigue_percent = int(self.fatigue_score)
        faces_text = "face" if self.detected_faces_count == 1 else "faces"
        self.fatigue_label.config(
            text=f"Mental Fatigue Level: {fatigue_percent}% ({self.detected_faces_count} {faces_text})"
        )
        self.fatigue_progress['value'] = fatigue_percent
        self.update_progress_bar_color(fatigue_percent)
        
        # Update emotions display
        self.emotions_text.delete(1.0, tk.END)
        
        if self.detected_faces_count == 0:
            self.emotions_text.insert(tk.END, "No faces detected in current frame")
        else:
            results_str = f"Detected {self.detected_faces_count} face(s):\n\n"
            
            # Show average emotions across all faces
            if all_emotions:
                avg_emotions = {}
                for emotion in ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']:
                    avg_emotions[emotion] = np.mean([face_emotions.get(emotion, 0) for face_emotions in all_emotions])
                
                results_str += "Average Emotions Across All Faces:\n"
                for emotion, score in avg_emotions.items():
                    results_str += f"  {emotion.capitalize()}: {score:.3f}\n"
                
                results_str += f"\nCalculated Mental Fatigue: {self.fatigue_score:.1f}%\n"
                
                if eye_openness is not None:
                    results_str += f"Average Eye Openness: {eye_openness:.1f}%\n"
                    if eye_openness < 30:
                        results_str += "⚠️ Very low eye openness detected - high fatigue indicator\n"
                    elif eye_openness < 50:
                        results_str += "⚠️ Low eye openness detected - moderate fatigue indicator\n"
                
                # Show individual face details if multiple faces
                if len(all_emotions) > 1:
                    results_str += f"\nIndividual Face Details:\n"
                    for i, emotions in enumerate(all_emotions, 1):
                        dominant = max(emotions, key=emotions.get)
                        results_str += f"  Face {i}: {dominant} ({emotions[dominant]:.3f})\n"
            
            self.emotions_text.insert(tk.END, results_str)
    
    def video_capture_loop(self):
        """Main video capture and processing loop"""
        while self.is_running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    if isinstance(self.video_source, str):  # Video file ended
                        print("Video file ended or cannot be read")
                        # Reset video to beginning for continuous playback
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        print("Failed to read from webcam")
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
    required_libs = [
        ('cv2', 'opencv-python'),
        ('PIL', 'Pillow'), 
        ('fer', 'fer'),
        ('numpy', 'numpy')
    ]
    
    optional_libs = [
        ('dlib', 'dlib'),
        ('face_recognition', 'face-recognition')
    ]
    
    missing_libs = []
    missing_optional = []
    
    for lib, install_name in required_libs:
        try:
            __import__(lib)
        except ImportError:
            missing_libs.append((lib, install_name))
    
    for lib, install_name in optional_libs:
        try:
            __import__(lib)
        except ImportError:
            missing_optional.append((lib, install_name))
    
    if missing_libs:
        print("Missing required libraries:")
        for lib, install_name in missing_libs:
            print(f"  Install {lib}: pip install {install_name}")
        sys.exit(1)
    
    if missing_optional:
        print("Missing optional libraries (eye detection will be disabled):")
        for lib, install_name in missing_optional:
            print(f"  Install {lib}: pip install {install_name}")
        print("The application will still work without eye detection.\n")
    
    # Create and run application
    root = tk.Tk()
    app = MentalFatigueDetector(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
