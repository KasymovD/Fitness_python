import warnings
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import threading

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

squat_counter = 0
squat_stage = None
landmarks_global = None


def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360.0 - angle

    return angle


def process_video():
    global squat_counter, squat_stage, landmarks_global

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Error: Could not read frame from video capture.")
            break

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            landmarks_global = results.pose_landmarks.landmark

            try:
                hip = [landmarks_global[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                       landmarks_global[mp_pose.PoseLandmark.RIGHT_HIP.value].y,
                       landmarks_global[mp_pose.PoseLandmark.RIGHT_HIP.value].z]
                knee = [landmarks_global[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                        landmarks_global[mp_pose.PoseLandmark.RIGHT_KNEE.value].y,
                        landmarks_global[mp_pose.PoseLandmark.RIGHT_KNEE.value].z]
                ankle = [landmarks_global[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                         landmarks_global[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y,
                         landmarks_global[mp_pose.PoseLandmark.RIGHT_ANKLE.value].z]

                squat_angle = calculate_angle(hip, knee, ankle)

                if squat_angle > 160:
                    squat_stage = "up"
                if squat_angle < 90 and squat_stage == 'up':
                    squat_stage = "down"
                    squat_counter += 1
                    print(f'Squats: {squat_counter}')

            except Exception as e:
                print(f"Error calculating angles: {e}")
                pass

            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            cv2.putText(image, 'Squats: ' + str(squat_counter),
                        (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('Squat Counter', image)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            print("User requested exit.")
            break

    cap.release()
    cv2.destroyAllWindows()


def update_plot(frame):
    global landmarks_global

    if landmarks_global:
        ax.clear()
        x_vals = [landmark.x for landmark in landmarks_global]
        y_vals = [landmark.y for landmark in landmarks_global]
        z_vals = [landmark.z for landmark in landmarks_global]

        ax.scatter(z_vals, x_vals, y_vals)

        for connection in mp_pose.POSE_CONNECTIONS:
            start = landmarks_global[connection[0]]
            end = landmarks_global[connection[1]]
            ax.plot([start.z, end.z], [start.x, end.x], [start.y, end.y], 'r-')

        ax.set_xlabel('Z')
        ax.set_ylabel('X')
        ax.set_zlabel('Y')
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([1, -1])

    plt.draw()


if __name__ == '__main__':
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ani = FuncAnimation(fig, update_plot, interval=100, cache_frame_data=False)

    video_thread = threading.Thread(target=process_video)
    video_thread.start()

    plt.show()
    video_thread.join()
