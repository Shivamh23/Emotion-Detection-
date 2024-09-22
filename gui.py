import tkinter as tk
from tkinter import filedialog
from tkinter import *
from tensorflow.keras.models import model_from_json
from PIL import Image, ImageTk
import numpy as np
import cv2

# Load model function
def facialExpressionModel(json_file, weight_file):
    with open(json_file, "r") as file:
        loaded_model_json = file.read()
        model = model_from_json(loaded_model_json)
        model.load_weights(weight_file)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Main Tkinter window
top = tk.Tk()
top.geometry('800x600')
top.title("Emotion Detector")
top.configure(background='#CDCDCD')

# Labels and placeholders for image and emotion text
label1 = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
sign_image = Label(top)

# Load the face detector
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load your trained model
model = facialExpressionModel("model_a1.json", "model_weights.weights.h5")

# List of emotion categories
EMOTIONS_LIST = ["Angry", "Sad", "Surprised", "Disgust", "Fear", "Neutral"]

# Function to detect emotions in the uploaded image
def Detect(file_path):
    global label_packed
    image = cv2.imread(file_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, 1.3, 5)

    try:
        for (x, y, w, h) in faces:
            fc = gray_image[y:y+h, x:x+w]
            roi = cv2.resize(fc, (48, 48))
            roi = roi[np.newaxis, :, :, np.newaxis]  # Reshape the region for model prediction
            pred = EMOTIONS_LIST[np.argmax(model.predict(roi))]
            label1.configure(foreground="#011638", text=f"Predicted Emotion: {pred}")
            print(f"Predicted Emotion: {pred}")
    except Exception as e:
        label1.configure(foreground="#011638", text="Unable to detect emotion")
        print(f"Error: {e}")

# Function to display "Detect Emotion" button after uploading an image
def show_Detect_Button(file_path):
    detect_b = Button(top, text="Detect Emotion", command=lambda: Detect(file_path), padx=10, pady=5)
    detect_b.configure(background="#364156", foreground='white', font=('arial', 10, 'bold'))
    detect_b.place(relx=0.79, rely=0.46)

# Function to upload an image
def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail((top.winfo_width() // 2.25, top.winfo_height() // 2.25))
        im = ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image = im
        label1.configure(text='')
        show_Detect_Button(file_path)
    except Exception as e:
        print(f"Error: {e}")
        pass

# Upload button
upload = Button(top, text="Upload Image", command=upload_image, padx=10, pady=5)
upload.configure(background="#364156", foreground="white", font=('arial', 20, 'bold'))
upload.pack(side="bottom", pady=50)

# Displaying uploaded image and emotion
sign_image.pack(side='bottom', expand='True')
label1.pack(side='bottom', expand='True')

# Heading
heading = Label(top, text="Emotion Detector", pady=20, font=('arial', 25, 'bold'))
heading.configure(background='#CDCDCD', foreground='#364156')
heading.pack()

# Run the main loop
top.mainloop()
