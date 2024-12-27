import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model

class CameraApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Camera App")
        self.model = load_model('face_recognition_model.h5', compile = False)
        self.IMAGE_SIZE = 128
        self.label_map = {'thuphuong': 0, 'Bui Minh Ngoc': 1, 'dieulinh': 2, 'lananh': 3}
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Buttons frame
        self.btn_frame = tk.Frame(window)
        self.btn_frame.pack(pady=10)

        # Create buttons
        self.camera_btn = tk.Button(self.btn_frame, text="Open Camera", command=self.open_camera)
        self.camera_btn.pack(side=tk.LEFT, padx=5)

        self.load_btn = tk.Button(self.btn_frame, text="Load Image", command=self.load_image)
        self.load_btn.pack(side=tk.LEFT, padx=5)

        # Image display label
        self.label = tk.Label(window)
        self.label.pack()

        self.cap = None
        self.is_camera_open = False

    def open_camera(self):
        if not self.is_camera_open:
            self.cap = cv2.VideoCapture(0)
            self.is_camera_open = True
            self.window.geometry("1000x800")
            self.update_camera()
            self.camera_btn.config(text="Close Camera")
        else:
            self.cap.release()
            self.is_camera_open = False
            self.label.config(image='')
            self.camera_btn.config(text="Open Camera")
            self.window.geometry("")

    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            face_roi = frame[y:y + h, x:x + w]
            face_img = cv2.resize(face_roi, (self.IMG_SIZE, self.IMG_SIZE))
            face_img = face_img / 255.0
            face_img = np.expand_dims(face_img, axis=0)

            predictions = self.model.predict(face_img)
            predicted_label = np.argmax(predictions)
            label_name = [name for name, label in self.label_map.items() if label == predicted_label][0]

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, label_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        return frame

    def update_camera(self):
        if self.is_camera_open:
            ret, frame = self.cap.read()
            if ret:
                frame = self.detect_faces(frame)

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                self.label.imgtk = imgtk
                self.label.configure(image=imgtk)
            self.window.after(10, self.update_camera)

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            img = Image.open(file_path)
            imgtk = ImageTk.PhotoImage(img)
            self.label.imgtk = imgtk
            self.label.configure(image=imgtk)

    # def __del__(self):
    #     if self.cap is not None:
    #         self.cap.release()

if __name__ == "__main__":
    root = tk.Tk()
    app = CameraApp(root)
    root.mainloop()