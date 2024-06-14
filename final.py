import cv2
import mediapipe as mp
import numpy as np
from fpdf import FPDF

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

push_up_counter = 0
push_up_stage = None
squat_counter = 0
squat_stage = None


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
    global push_up_counter, push_up_stage, squat_counter, squat_stage

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark

            shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

            push_up_angle = calculate_angle(shoulder, elbow, wrist)
            squat_angle = calculate_angle(hip, knee, ankle)

            if push_up_angle > 160:
                push_up_stage = "up"
            if push_up_angle < 90 and push_up_stage == 'up':
                push_up_stage = "down"
                push_up_counter += 1
                print(f'Push ups: {push_up_counter}')

            if squat_angle > 160:
                squat_stage = "up"
            if squat_angle < 90 and squat_stage == 'up':
                squat_stage = "down"
                squat_counter += 1
                print(f'Squats: {squat_counter}')

        except:
            pass

        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.putText(image, 'Push ups: ' + str(push_up_counter),
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.putText(image, 'Squats: ' + str(squat_counter),
                    (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('Exercise Counter', image)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    create_pdf(push_up_counter, squat_counter)


class PDF(FPDF):
    def header(self):
        # Добавление изображения в заголовок
        self.image('logo_170X115.png', 10, 8, 33)
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'SPORT RESULTS', 0, 1, 'C')

    def add_certificate_content(self, push_up_count, squat_count):
        self.set_font('Arial', '', 12)
        self.cell(0, 10, '', 0, 1)  # Spacer
        self.cell(0, 10, '', 0, 1)  # Spacer
        self.cell(0, 10, '', 0, 1)  # Spacer
        self.cell(30, 10, 'Student Number:', 0, 0)
        self.cell(80, 10, '', 1, 1)

        self.cell(0, 20, '', 0, 1)  # Spacer
        self.multi_cell(0, 10, "I, the undersigned Dr___________________________, Doctor of Medicine,\n"
                               "Certify that the examination of Mr/Ms__________________________________\n"
                               "Date of birth: _______________________ Age: ___________________\n"
                               "reveals no contraindications for participating in running competitions.\n\n"
                               "Medical certificate issued in (place):_________________________")

        self.cell(0, 20, '', 0, 1)  # Spacer
        self.cell(40, 10, 'Date:', 0, 0)
        self.cell(60, 10, '', 1, 0)
        self.cell(50, 10, 'Doctors sign:', 0, 0)
        self.cell(40, 10, '', 1, 1)

        self.cell(0, 20, 'Doctors Stamp:', 0, 1, 'C')

        # Добавление данных о упражнениях
        self.cell(0, 20, '', 0, 1)  # Spacer
        self.cell(200, 10, f'Total Push Ups: {push_up_count}', 0, 1, 'L')
        self.cell(200, 10, f'Total Squats: {squat_count}', 0, 1, 'L')


def create_pdf(push_up_count, squat_count):
    pdf = PDF()
    pdf.add_page()
    pdf.add_certificate_content(push_up_count, squat_count)
    pdf.output("sport_results.pdf")


if __name__ == '__main__':
    process_video()
