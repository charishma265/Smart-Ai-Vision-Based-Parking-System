
import os

# FIX for Keras model loading
from flask import Flask, render_template, request
import cv2
import numpy as np
import tensorflow as tf
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
ROI_FOLDER = "static/rois"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ROI_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# -------------------------------
# Load trained CNN model
# -------------------------------
from tensorflow.keras.models import load_model
model = load_model(r"C:\Users\ADMIN\Downloads\realtimeparking\realtimeparking\cnn_model.h5",compile=False)
# -------------------------------
# Image preprocessing
# -------------------------------
def preprocess_image(image):
    image = cv2.resize(image, (128, 128))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image / 255.0
    image = image.reshape(128, 128, 1)
    return np.expand_dims(image, axis=0)


# -------------------------------
# Predict parked / empty
# -------------------------------
def predict_image(image):
    prediction = model.predict(preprocess_image(image), verbose=0)
    return "parked" if prediction > 0.5 else "empty"


# -------------------------------
# Process video & compute stats
# -------------------------------
def process_video(video_path):

    # clear old roi images
    for f in os.listdir(ROI_FOLDER):
        os.remove(os.path.join(ROI_FOLDER, f))

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    interval_sec = 2
    frame_interval = fps * interval_sec

    cell_width = 100
    cell_height = 60
    columns_of_interest = [2, 3]

    slot_state = {}
    total_park_time = {}

    roi_files = []
    logs = []

    frame_count = 0
    roi_index = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:

            current_time = frame_count // fps
            height, width, _ = frame.shape
            slot_no = 1

            for y in range(0, height, cell_height):
                for x in range(0, width, cell_width):

                    col_label = (x // cell_width) + 1

                    if col_label in columns_of_interest:
                        if y + cell_height <= height and x + cell_width <= width:

                            slot_img = frame[y:y + cell_height, x:x + cell_width]
                            status = predict_image(slot_img)

                            if slot_no not in slot_state:
                                slot_state[slot_no] = {
                                    "status": "empty",
                                    "entry_time": None
                                }
                                total_park_time[slot_no] = 0

                            prev_status = slot_state[slot_no]["status"]

                            # ENTRY
                            if prev_status == "empty" and status == "parked":
                                slot_state[slot_no]["entry_time"] = current_time
                                logs.append(
                                    f"Time {current_time}s | Slot {slot_no} | Car ENTERED"
                                )

                            # EXIT
                            if prev_status == "parked" and status == "empty":
                                entry_time = slot_state[slot_no]["entry_time"]
                                parked_time = current_time - entry_time
                                total_park_time[slot_no] += parked_time

                                logs.append(
                                    f"Slot {slot_no} PARKED for {parked_time}s"
                                )

                                slot_state[slot_no]["entry_time"] = None

                            slot_state[slot_no]["status"] = status

                            logs.append(
                                f"Time {current_time}s | Slot {slot_no} | {status}"
                            )

                            # Save ROI image
                            roi_rgb = cv2.cvtColor(slot_img, cv2.COLOR_BGR2RGB)

                            filename = f"roi_{roi_index}_slot{slot_no}_{status}_{current_time}s.jpg"
                            save_path = os.path.join(ROI_FOLDER, filename)

                            cv2.imwrite(save_path, cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2BGR))

                            roi_files.append(filename)

                            roi_index += 1
                            slot_no += 1

        frame_count += 1

    cap.release()

    # -------------------------------
    # Close parking at video end
    # -------------------------------
    video_end_time = frame_count // fps

    for slot, state in slot_state.items():
        if state["status"] == "parked" and state["entry_time"] is not None:
            remaining_time = video_end_time - state["entry_time"]
            total_park_time[slot] += remaining_time
            logs.append(
                f"Slot {slot} PARKED till end for {remaining_time}s"
            )

    # -------------------------------
    # Final bill + dashboard stats
    # -------------------------------
    logs.append("\n----- FINAL BILL SUMMARY -----")

    total_revenue = 0

    for slot, time_sec in total_park_time.items():
        bill = (time_sec // 4) * 5
        total_revenue += bill
        logs.append(
            f"Slot {slot}: Parked Time = {time_sec}s | Bill = ₹{bill}"
        )

    total_slots = len(total_park_time)
    occupied_slots = sum(
        1 for s in slot_state.values() if s["status"] == "parked"
    )
    empty_slots = max(total_slots - occupied_slots, 0)
    total_parked_time = sum(total_park_time.values())

    stats = {
        "total_slots": total_slots,
        "occupied_slots": occupied_slots,
        "empty_slots": empty_slots,
        "total_revenue": total_revenue,
        "total_parked_time": total_parked_time,
    }

    return roi_files, "\n".join(logs), stats


# -------------------------------
# Routes
# -------------------------------
@app.route("/", methods=["GET", "POST"])
def index():

    roi_files = []
    logs = ""
    video_filename = None
    stats = {
        "total_slots": 0,
        "occupied_slots": 0,
        "empty_slots": 0,
        "total_revenue": 0,
        "total_parked_time": 0,
    }

    if request.method == "POST":

        file = request.files["video"]
        filename = secure_filename(file.filename)
        video_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(video_path)
        video_filename = filename

        roi_files, logs, stats = process_video(video_path)

    return render_template(
        "index.html",
        roi_files=roi_files,
        logs=logs,
        stats=stats,
        video_filename=video_filename,
    )


if __name__ == "__main__":
    app.run(debug=True)