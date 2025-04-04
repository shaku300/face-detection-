import cv2
import time
import numpy as np
import os

# Open the laptop camera (0 for default webcam)
cap = cv2.VideoCapture(0)

# Get the original FPS of the camera (fallback to 30 if unknown)
original_fps = int(cap.get(cv2.CAP_PROP_FPS))
if original_fps == 0:  # Some cameras may return 0 for FPS
    original_fps = 29
    

low_fps = 1  # Lower FPS when no motion
target_fps = original_fps  # Start with original FPS

# Add downsampling factor for low resolution
downscale_factor = 4  # Will reduce resolution by 4x when no motion

# Create output directory if it doesn't exist
output_dir = "comparison_recordings"
os.makedirs(output_dir, exist_ok=True)

# Get timestamp for unique filenames
timestamp = time.strftime("%Y%m%d_%H%M%S")

# Initialize video writers with compression
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_size = (frame_width, frame_height)

# Using H.264 codec for better compression
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Create two video writers - one for original and one for adaptive footage
original_output = cv2.VideoWriter(
    f'{output_dir}/original_{timestamp}.mp4',
    fourcc,
    original_fps,
    frame_size,
    isColor=True
)

adaptive_output = cv2.VideoWriter(
    f'{output_dir}/adaptive_{timestamp}.mp4',
    fourcc,
    original_fps,  # We'll control FPS through frame writing
    frame_size,
    isColor=True
)

print(f"🎥 Starting camera... Original FPS detected: {original_fps}")
print(f"📁 Saving recordings to {output_dir}/")

# Initialize KNN background subtractor
bg_subtractor = cv2.createBackgroundSubtractorKNN()

# Motion detection variables
motion_detected = True  # Assume motion is present at start
last_motion_time = time.time()  # Track last motion time
room_status = "Occupied"  # Initial status
frame_count = 0  # Counter for FPS control

while cap.isOpened():
    start_time = time.time()  # Track time for FPS control

    ret, frame = cap.read()
    if not ret:
        print("❌ Error: Could not read frame from camera.")
        break

    # Always write original frame to original output
    original_output.write(frame)
    
    # Create adaptive frame based on motion detection
    adaptive_frame = frame.copy()
    
    # Apply KNN background subtraction
    fg_mask = bg_subtractor.apply(frame)

    # Morphological transformations
    kernel = np.ones((5, 5), np.uint8)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_DILATE, kernel)

    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    motion_score = sum(cv2.contourArea(c) for c in contours)

    # Handle motion detection and frame processing
    if motion_score > 1000:
        if not motion_detected:
            print("⚡ Movement detected.... using original FPS and resolution")
            motion_detected = True
        last_motion_time = time.time()
        target_fps = original_fps
        room_status = "Occupied"
        
        # Write frame at original quality
        adaptive_output.write(adaptive_frame)
        frame_count = 0
        
        # Draw bounding boxes
        for contour in contours:
            if cv2.contourArea(contour) > 500:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(adaptive_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    else:
        if time.time() - last_motion_time > 5:
            if motion_detected:
                print("⏳ No movement detected.... using low FPS and resolution")
                motion_detected = False
                room_status = "Unoccupied"
            target_fps = low_fps
            
            # Reduce quality for adaptive frame
            small_frame = cv2.resize(adaptive_frame, 
                                   (frame_width // downscale_factor, 
                                    frame_height // downscale_factor))
            adaptive_frame = cv2.resize(small_frame, (frame_width, frame_height))
            
            # Write reduced quality frame at lower FPS
            frame_count += 1
            if frame_count >= (original_fps // low_fps):
                adaptive_output.write(adaptive_frame)
                frame_count = 0

    # Display status on frame
    cv2.putText(adaptive_frame, f"Room Status: {room_status}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Show the live camera feed
    cv2.imshow('Live Camera Feed', adaptive_frame)
    
    # Press 'q' to quit
    key = cv2.waitKey(1) & 0xFF
    if key == ord('g'):
        print("🔚 Exiting...")
        break

    # Control frame rate
    elapsed_time = time.time() - start_time
    sleep_time = max(1.0 / target_fps - elapsed_time, 0)
    time.sleep(sleep_time)

# Cleanup
cap.release()
original_output.release()
adaptive_output.release()
cv2.destroyAllWindows()

# Print file sizes for comparison
original_size = os.path.getsize(f'{output_dir}/original_{timestamp}.mp4')
adaptive_size = os.path.getsize(f'{output_dir}/adaptive_{timestamp}.mp4')
saved_space = original_size - adaptive_size
saved_percentage = (saved_space / original_size) * 100

print("\n📊 Storage Comparison:")
print(f"Original file size: {original_size / (1024*1024):.2f} MB")
print(f"Adaptive file size: {adaptive_size / (1024*1024):.2f} MB")
print(f"Saved space: {saved_space / (1024*1024):.2f} MB ({saved_percentage:.1f}%)")
