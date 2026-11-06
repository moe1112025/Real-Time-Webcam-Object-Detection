import cv2
import time
from ultralytics import YOLO

# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================
MODEL_PATH = "yolov8n.pt"              # Pretrained YOLOv8 Nano model weights file [Fixed typo from yolov8.pt]
CAMERA_INDEX = 0                       # System index for the default webcam
CONFIDENCE_THRESHOLD = 0.40            # Minimum prediction confidence score filter [Fixed spelling]
IMAGE_SIZE = 640                       # Image size resolution passed to the YOLO model


# ==========================================
# 2. REAL-TIME DETECTION PIPELINE
# ==========================================
def main():
    """
    Initializes the YOLOv8 object detection model, opens the webcam stream, 
    processes frames in real-time with bounding boxes, computes FPS, and handles graceful shutdown.
    """
    print("========================================")
    print("      REAL-TIME WEBCAM OBJECT DETECTION ")
    print("========================================")

    # Load the pretrained YOLOv8 model instance
    print("[INFO] Loading YOLOv8 model...")
    model = YOLO(MODEL_PATH)

    # Open the default system webcam video capture stream
    print("[INFO] Starting webcam...")
    camera = cv2.VideoCapture(CAMERA_INDEX)

    # Verify that the webcam stream opened successfully
    if not camera.isOpened():
        raise FileNotFoundError("Could not open webcam device.")

    print("[INFO] Webcam started successfully.")
    print("[INFO] Press 'Q' to exit.")

    # Initialize timestamp tracker for frame rate (FPS) calculations
    previous_time = time.time()

    try:
        # Start the continuous real-time video processing loop
        while True:
            # Capture a single frame from the webcam stream
            success, frame = camera.read()

            # Handle camera read failures gracefully
            if not success or frame is None:
                print("[WARNING] Failed to read webcam frame.")
                break

            # Run YOLOv8 object detection inference on the captured frame
            results = model.predict(
                source = frame,
                conf = CONFIDENCE_THRESHOLD,
                imgsz = IMAGE_SIZE,
                verbose = False
            )

            # Automatically draw bounding boxes, class labels, and confidence scores onto the frame
            output_frame = results[0].plot()

            # Calculate instantaneous frame rate (FPS)
            current_time = time.time()
            elapsed_time = current_time - previous_time
            fps = 1.0 / elapsed_time if elapsed_time > 0 else 0.0
            previous_time = current_time

            # Render the FPS counter text overlay onto the top-left of the output frame
            cv2.putText(
                output_frame, 
                f"FPS: {fps:.1f}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0, 
                (0, 255, 0), 
                2, 
                cv2.LINE_AA
            )

            # Display the processed video frame inside an interactive OpenCV window
            cv2.imshow("YOLOv8 Real-Time Detection", output_frame)

            # Capture keyboard input (wait 1ms per frame) and check if 'q' or 'Q' is pressed
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                print("[INFO] Shutdown requested by user.")
                break

    finally:
        # ==========================================
        # 3. RESOURCE CLEANUP
        # ==========================================
        # Release the webcam capture hardware and close all active OpenCV windows
        camera.release()
        cv2.destroyAllWindows()

        print("\n[INFO] Webcam stopped.")
        print("[SUCCESS] Application closed successfully.")


# ==========================================
# 4. SCRIPT ENTRY POINT
# ==========================================
if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # Catch and print any runtime exceptions cleanly [Fixed stray syntax error]
        print(f"\n[ERROR] Application failed: {error}")
        raise