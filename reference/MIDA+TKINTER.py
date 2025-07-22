from pathlib import Path
from tkinter import Tk, Canvas, Button, PhotoImage
import threading
import cv2
import dlib
import numpy as np
from scipy.spatial import distance
import time
import pygame
import logging

logging.basicConfig(level=logging.WARNING)

pygame.init()
pygame.mixer.init()
warning_sound = pygame.mixer.Sound("music.wav")

shape_predictor_path = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor_path)

is_sound_playing = False  


def get_eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


def play_warning_sound():
    """
    Plays the warning sound for a limited time in a non-blocking manner.
    """
    global is_sound_playing
    if not is_sound_playing:  
        is_sound_playing = True
        warning_sound.play()
        time.sleep(0.4)  
        warning_sound.stop()
        is_sound_playing = False


def drowsiness_detection():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        logging.error("Unable to open webcam.")
        return

    EAR_THRESHOLD = 0.15  
    EAR_CONSEC_FRAMES = 15
    COUNTER = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            logging.error("Failed to capture frame.")
            break

        if frame is None or frame.dtype != np.uint8:
            logging.error("Frame is not in the expected 8-bit format.")
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if gray.dtype != np.uint8:
            logging.error("Grayscale image is not 8-bit.")
            continue

        faces = detector(gray)

        for face in faces:
            landmarks = predictor(gray, face)
            landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])

            left_eye = landmarks[36:42]
            right_eye = landmarks[42:48]

            left_ear = get_eye_aspect_ratio(left_eye)
            right_ear = get_eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0

            if ear < EAR_THRESHOLD:
                COUNTER += 1
                if COUNTER >= EAR_CONSEC_FRAMES and not is_sound_playing:
                    threading.Thread(target=play_warning_sound).start()
                    cv2.putText(frame, "DROWSINESS DETECTED!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                COUNTER = 0

            cv2.polylines(frame, [left_eye], True, (0, 255, 0), 1)
            cv2.polylines(frame, [right_eye], True, (0, 255, 0), 1)

        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def start_detection():
    detection_thread = threading.Thread(target=drowsiness_detection)
    detection_thread.start()


def create_gui():
    OUTPUT_PATH = Path(__file__).parent
    ASSETS_PATH = OUTPUT_PATH / Path(r"C:\Users\SMSAINSSULTANHAJIAHM\Desktop\build\assets\frame0")

    def relative_to_assets(path: str) -> Path:
        return ASSETS_PATH / Path(path)

    window = Tk()
    window.geometry("418x875")
    window.configure(bg="#FFFFFF")

    canvas = Canvas(
        window,
        bg="#FFFFFF",
        height=875,
        width=418,
        bd=0,
        highlightthickness=0,
        relief="ridge"
    )
    canvas.place(x=0, y=0)

    image_image_1 = PhotoImage(file=relative_to_assets("image_1.png"))
    canvas.create_image(209.0, 437.0, image=image_image_1)

    image_image_2 = PhotoImage(file=relative_to_assets("image_2.png"))
    canvas.create_image(209.0, 352.0, image=image_image_2)

    button_image_1 = PhotoImage(file=relative_to_assets("button_1.png"))
    button_1 = Button(
        image=button_image_1,
        borderwidth=0,
        highlightthickness=0,
        command=start_detection,
        relief="flat"
    )
    button_1.place(
        x=83.0,
        y=514.0,
        width=250.0,
        height=90.0
    )

    window.resizable(False, False)
    window.mainloop()


if __name__ == "__main__":
    create_gui()
