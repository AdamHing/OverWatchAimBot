# import threading
# from ultralytics import YOLO
# import cv2
# # Load the models
# model1 = YOLO('yolov8n.pt')
# model2 = YOLO('yolov8m-pose.pt')

# # Function to run tracking
# def track_video(model, source, conf, show, save):
#     results = model.track(source=source, show=show, conf=conf, save=save)
#     annotated_image = results[0].plot()
#     annotated_image = cv2.resize(annotated_image,(900,900))
#     cv2.imshow("screen", annotated_image)

#     # Process results here or return them
# vid = r"D:\Coding\Projects\OverWatchCV\Walking Shibuya Crossing at Night, Binaural City Sounds in Tokyo _ 4k.mp4"
# # Create threads for each model
# thread1 = threading.Thread(target=track_video, args=(model1, vid, 0.3, False, False))
# thread2 = threading.Thread(target=track_video, args=(model2, vid, 0.3, False, False))

# # Start the threads
# thread1.start()
# thread2.start()

# # Wait for both threads to finish
# thread1.join()
# thread2.join()




import threading
from ultralytics import YOLO
import cv2
# Load the models
model1 = YOLO('yolov8n.pt')
model2 = YOLO('yolov8l-pose.pt')

# Function to run tracking
def track_video(model, source, conf, show, save):
    results = model.track(source=source, show=show, conf=conf, save=save)
    annotated_image = results[0].plot()
    annotated_image = cv2.resize(annotated_image,(900,900))
    # cv2.imshow("screen", annotated_image)

    # Process results here or return them 
vid = r"D:\Coding\Projects\OverWatchCV\Overwatch 2 Gameplay (No Commentary).mp4"

track_video(model2, vid, 0.2, True, False)

# Create threads for each model
# thread1 = threading.Thread(target=track_video, args=(model1, vid, 0.3, False, False))
# thread2 = threading.Thread(target=track_video, args=(model2, vid, 0.3, False, False))

# Start the threads
# thread1.start()
# thread2.start()

# # Wait for both threads to finish
# thread1.join()
# thread2.join()