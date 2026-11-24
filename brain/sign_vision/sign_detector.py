"""
Sign Detection Controller Module - Manages traffic sign detection using YOLO.

Following SRP: This module handles sign detection and provides debug images for streaming.
"""

import threading
import time
import sys
import os
from pathlib import Path
import cv2
import torch
from ultralytics import YOLO

# Import VideoStreamer from camera module (shared camera resource)
sys.path.insert(0, str(Path(__file__).parent.parent))
from camera.video_streamer import VideoStreamer


class SignDetector:
    """Manages traffic sign detection using YOLO model."""
    
    def __init__(self, video_streamer: VideoStreamer, 
                 model_path: str = None, 
                 confidence_threshold: float = 0.6,
                 source: str = None):
        """
        Initialize the sign detection controller.
        
        Args:
            video_streamer: VideoStreamer instance for getting frames
            model_path: Path to YOLO model file (default: sign_vision/weights/best.pt)
            confidence_threshold: Minimum confidence for detections (default: 0.6)
            source: Optional source ID (camera index or video path) to use a dedicated camera.
                    If None, uses the shared video_streamer.
        """
        self.video_streamer = video_streamer
        self.confidence_threshold = confidence_threshold
        self.source = source
        self.cap = None
        
        # Determine model path
        if model_path is None:
            script_dir = Path(__file__).parent
            #model_path = script_dir / "weights" / "best.pt"
            model_path = script_dir / "weights" / "best.engine"
        self.model_path = str(model_path)
        self.model = None
        self.device = None
        
        self.is_running = False
        self.thread = None
        self.lock = threading.Lock()
        
        # Statistics
        self.detection_count = 0
        self.frame_count = 0
        self.error_count = 0
        
        # Debug images storage
        self.last_detection_image = None
        self.last_detections = []  # List of detected signs with info
    
    def _detect_and_configure_gpu(self):
        """Detect GPU and configure for maximum performance."""
        if torch.cuda.is_available():
            device = "cuda"
            torch.backends.cudnn.benchmark = True
            return device
        return "cpu"
    
    def _load_model(self):
        """Load YOLO model, prioritizing engine if exists."""
        if not os.path.exists(self.model_path):
            print(f"[SignDetector] Error: Model not found: {self.model_path}")
            return None
        
        engine_path = self.model_path.replace('.pt', '.engine')
        print(f"[SignDetector] model_path: {self.model_path}")
        print(f"[SignDetector] engine_path: {engine_path}, exists? {os.path.exists(engine_path)}")
        
        # Prioritize engine if exists
        if os.path.exists(engine_path):
            print(f"[SignDetector] Engine file found: {engine_path}")
            print("[SignDetector] Using TensorRT optimized model")
            model = YOLO(engine_path, task='detect')
            return model
        
        # Load .pt and export to engine if GPU available
        model = YOLO(self.model_path, task='detect')
        if self.device == "cuda":
            model.to(self.device)
            if not os.path.exists(engine_path):
                try:
                    model.export(format='engine')
                    if os.path.exists(engine_path):
                        # Reload as engine (doesn't need .to(device))
                        model = YOLO(engine_path, task='detect')
                except Exception as e:
                    print(f"[SignDetector] Warning: Could not export to engine: {e}")
                    # Continue with .pt if export fails
        
        return model
    
    def initialize(self) -> bool:
        """
        Initialize the sign detection controller.
        
        Returns:
            True if initialized successfully, False otherwise
        """
        try:
            # Configure GPU
            self.device = self._detect_and_configure_gpu()
            print(f"[SignDetector] Using device: {self.device}")
            
            # Load model
            self.model = self._load_model()
            if self.model is None:
                return False
            
            print(f"[SignDetector] Model loaded successfully")

            # Warm-up: the TensorRT engine is really loaded on the FIRST inference, which
            # freezes the whole Python process for seconds. Do it here (during init, while
            # the dashboard keeps the ESP32 in MANUAL) instead of in the detection loop,
            # where the frozen heartbeat made the ESP32 watchdog go to FAULT.
            try:
                import numpy as np
                t0 = time.time()
                self.model(np.zeros((480, 640, 3), dtype=np.uint8), verbose=False, device=self.device)
                print(f"[SignDetector] Warm-up inference done in {time.time() - t0:.1f}s")
            except Exception as e:
                print(f"[SignDetector] Warm-up inference failed (non fatal): {e}")

            print(f"[SignDetector] Confidence threshold: {self.confidence_threshold}")
            return True
            
        except Exception as e:
            print(f"[SignDetector] Error initializing: {e}")
            return False
    
    def start(self):
        """Start the sign detection controller."""
        with self.lock:
            if self.is_running:
                return False
            
            if self.model is None:
                if not self.initialize():
                    return False
            
            self.is_running = True
            self.last_detection_image = None  # Clear stale image
            self.thread = threading.Thread(target=self._detection_loop, daemon=True)
            self.thread.start()
            print("[SignDetector] Started")
            return True
    
    def stop(self):
        """Stop the sign detection controller."""
        with self.lock:
            if not self.is_running:
                return False
            
            self.is_running = False
            
            # Release dedicated camera if used
            if self.cap is not None:
                self.cap.release()
                self.cap = None
                
            print("[SignDetector] Stopped")
            return True
    
    def _detection_loop(self):
        """Main detection loop running in background thread."""
        
        # If using dedicated source, open it
        if self.source is not None:
            try:
                # Try to convert to int if it's a number (camera index)
                source_id = int(self.source)
            except ValueError:
                source_id = self.source
                
            print(f"[SignDetector] Opening dedicated video source: {source_id}")
            self.cap = cv2.VideoCapture(source_id)
            if not self.cap.isOpened():
                print(f"[SignDetector] Error: Could not open video source {source_id}")
                # Fallback to shared streamer? Or stop?
                # For now, we'll just loop and try to reopen or fail
        
        while self.is_running:
            try:
                frame = None
                
                # Get frame from dedicated source if configured
                if self.cap is not None and self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if not ret:
                        # End of video or camera disconnected
                        print("[SignDetector] dedicated source ended or disconnected. Reopening...")
                        self.cap.release()
                        time.sleep(1.0)
                        try:
                            source_id = int(self.source) if str(self.source).isdigit() else self.source
                            self.cap = cv2.VideoCapture(source_id)
                        except:
                            pass
                        continue
                else:
                    # Use shared streamer
                    frame = self.video_streamer.get_frame()
                
                if frame is None:
                    time.sleep(0.033)  # Wait ~30ms if no frame
                    continue
                
                self.frame_count += 1
                
                # Run inference
                results = self.model(frame, verbose=False, device=self.device)[0]
                
                # Process detections
                detections = []
                detection_image = frame.copy()
                
                if results.boxes is not None and len(results.boxes) > 0:
                    boxes = results.boxes
                    for i in range(len(boxes)):
                        confidence = float(boxes.conf[i])
                        if confidence >= self.confidence_threshold:
                            # Get bounding box coordinates
                            box = boxes.xyxy[i].cpu().numpy()  # [x1, y1, x2, y2]
                            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                            
                            # Calculate center position
                            cx = int((x1 + x2) / 2)
                            cy = int((y1 + y2) / 2)
                            
                            # Get distance if available (from RealSense)
                            distance = None
                            if hasattr(self.video_streamer, 'get_distance') and self.source is None:
                                distance = self.video_streamer.get_distance(cx, cy)
                                # if distance is None:
                                #     print(f"[SignDetector] Warning: get_distance returned None for ({cx}, {cy})")
                            # else:
                            #     print("[SignDetector] Video streamer does not support get_distance")
                            
                            cls = int(boxes.cls[i])
                            class_name = self.model.names[cls]
                            
                            # Store detection info
                            detection_info = {
                                'class': class_name,
                                'confidence': confidence,
                                'bbox': [x1, y1, x2, y2],
                                'center': (cx, cy)
                            }
                            
                            if distance is not None:
                                detection_info['distance'] = distance
                                
                            detections.append(detection_info)
                            
                            # Draw bounding box
                            cv2.rectangle(detection_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            
                            # Draw center point
                            cv2.circle(detection_image, (cx, cy), 5, (0, 0, 255), -1)
                            
                            # Draw label with background and position
                            label = f"{class_name} {confidence:.1%} ({cx}, {cy})"
                            if distance is not None:
                                label += f" {distance:.2f}m"
                                
                            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                            label_y = y1 - 10 if y1 - 10 > 10 else y1 + 20
                            cv2.rectangle(detection_image, (x1, label_y - label_size[1] - 5), 
                                        (x1 + label_size[0], label_y + 5), (0, 255, 0), -1)
                            cv2.putText(detection_image, label, (x1, label_y), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                            
                            self.detection_count += 1
                
                # Store results
                with self.lock:
                    self.last_detection_image = detection_image
                    self.last_detections = detections
                
                # Control loop rate (~30 FPS)
                time.sleep(0.033)
                
            except Exception as e:
                print(f"[SignDetector] Error in detection loop: {e}")
                self.error_count += 1
                time.sleep(0.1)
    
    def get_status(self) -> dict:
        """Get current status of the sign detection controller."""
        with self.lock:
            return {
                'is_running': self.is_running,
                'detection_count': self.detection_count,
                'frame_count': self.frame_count,
                'error_count': self.error_count,
                'confidence_threshold': self.confidence_threshold,
                'current_detections': len(self.last_detections),
                'device': self.device
            }
    
    def get_detection_image(self):
        """
        Get the current detection image with bounding boxes.
        
        Returns:
            Image (numpy array) or None if not available
        """
        with self.lock:
            if self.last_detection_image is not None:
                return self.last_detection_image.copy()
        return None
    
    def get_detections(self) -> list:
        """
        Get list of current detections.
        
        Returns:
            List of detection dictionaries with 'class', 'confidence', and 'bbox'
        """
        with self.lock:
            return self.last_detections.copy()
    
    def update_confidence_threshold(self, threshold: float):
        """
        Update confidence threshold dynamically.
        
        Args:
            threshold: New confidence threshold (0.0 to 1.0)
        """
        if 0.0 <= threshold <= 1.0:
            with self.lock:
                self.confidence_threshold = threshold
            print(f"[SignDetector] Confidence threshold updated to {threshold}")
