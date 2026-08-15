# 🚗 SmartParkAI – Vision-Based Smart Parking Detection System

> **AI-powered parking occupancy detection and analytics using Computer Vision, OpenCV, TensorFlow CNN, and Flask.**

SmartParkAI is a vision-based smart parking system that analyzes CCTV or parking-lot video footage to identify **occupied and available parking spaces**. The system combines a trained **Convolutional Neural Network (CNN)** with **OpenCV-based video processing** to classify parking regions and generate useful parking analytics.

The application provides an interactive web dashboard where users can upload parking footage, process the video, monitor parking occupancy, view detected parking regions, track parking events, calculate parking duration, and estimate parking revenue.

---

## 🌟 Project Highlights

* 🎥 **CCTV / Parking Video Processing**
* 🧠 **CNN-based parking-space classification**
* 👁️ **OpenCV video and image processing**
* 🅿️ **Occupied vs Available slot detection**
* 📊 **Real-time-style parking dashboard**
* 📈 **Parking analytics and revenue visualization**
* 🚘 **Vehicle entry and exit event tracking**
* ⏱️ **Parking duration calculation**
* 💰 **Automatic parking billing calculation**
* 🖼️ **ROI snapshots for detected parking regions**
* 🌐 **Flask web application**
* 📱 **Responsive dashboard UI**
* 📋 **Parking event logs**
* 🧪 **CNN model loading/testing support**

---

# 🎯 Problem Statement

Finding available parking spaces can be difficult in busy parking areas. Drivers may spend considerable time searching for vacant spaces, while parking operators may lack a simple way to monitor occupancy and parking usage.

Traditional parking systems often depend on physical sensors installed at individual parking spaces.

SmartParkAI explores a **camera-based alternative** where existing parking/CCTV video can be analyzed using Computer Vision and Deep Learning.

The system identifies parking regions from video frames and uses a CNN model to determine whether each region is:

```text
🟢 EMPTY
🔴 PARKED
```

The detected information is then converted into parking statistics and analytics.

---

# 💡 Solution

SmartParkAI follows a computer-vision pipeline:

```text
              CCTV / Parking Video
                       │
                       ▼
                Flask Web App
                       │
                       ▼
                OpenCV Processing
                       │
                       ▼
              Frame Extraction
                       │
                       ▼
          Parking Region / ROI Creation
                       │
                       ▼
             Image Preprocessing
             128 × 128 Grayscale
                       │
                       ▼
             TensorFlow CNN Model
                       │
              ┌────────┴────────┐
              ▼                 ▼
          🅿️ PARKED          🟢 EMPTY
              │                 │
              └────────┬────────┘
                       ▼
              Event & State Tracking
                       │
             ┌─────────┼─────────┐
             ▼         ▼         ▼
          Duration   Billing   Statistics
             │         │         │
             └─────────┼─────────┘
                       ▼
             SmartParkAI Dashboard
```

---

# ⚙️ How the System Works

## 1. Upload Parking Video

The user uploads CCTV or parking-lot footage through the Flask web interface.

The dashboard supports video upload and preview before processing.

---

## 2. Video Processing

OpenCV reads the uploaded video frame by frame.

The application samples the video at a fixed interval rather than processing every frame continuously.

The current implementation processes frames approximately every **2 seconds**.

---

## 3. Parking Region Detection

The current implementation divides each frame into grid cells.

The configured cell dimensions are:

```text
Width  = 100 pixels
Height = 60 pixels
```

Selected columns are then treated as parking regions/ROIs.

---

## 4. Image Preprocessing

Each parking region is prepared before being passed to the CNN model.

The preprocessing pipeline is:

```text
Original ROI
     ↓
Resize to 128 × 128
     ↓
Convert BGR → Grayscale
     ↓
Normalize pixel values
     ↓
Reshape to 128 × 128 × 1
     ↓
Add batch dimension
     ↓
CNN Model
```

The implementation normalizes pixel values to the `0–1` range before prediction.

---

## 5. CNN Prediction

The TensorFlow/Keras CNN model predicts the parking-space status.

The current classification logic uses a threshold of `0.5`:

```text
Prediction > 0.5
        ↓
     PARKED

Prediction ≤ 0.5
        ↓
      EMPTY
```

---

# 🚘 Parking Event Tracking

SmartParkAI maintains the state of each detected parking region.

It identifies events such as:

```text
EMPTY → PARKED
```

### Vehicle Entry

When a previously empty slot becomes parked:

```text
Car ENTERED
```

The system records the corresponding entry time.

### Vehicle Exit

When a parked slot becomes empty:

```text
Car EXITED
```

The system calculates how long the vehicle remained parked.

The application also handles vehicles that remain parked until the end of the uploaded video.

---

# 💰 Parking Billing

The application calculates a parking bill based on the recorded parking duration.

The current implementation uses:

```text
Every 4 seconds → ₹5
```

The calculation is implemented as:

```python
bill = (time_sec // 4) * 5
```

This is a **demo billing configuration** and can be changed to a real-world hourly/minute-based parking tariff.

---

# 📊 Dashboard

The SmartParkAI dashboard provides an overview of the processed parking session.

It displays metrics including:

* Total parking slots
* Occupied slots
* Available slots
* Total parking duration
* Total revenue
* Uploaded video
* Parking event information

The dashboard also provides a CCTV/video preview and a video-processing interface.

---

# 📈 Parking Analytics

The analytics section provides a control-center style view of the processed parking session.

It includes:

### 💰 Revenue Visualization

A weekly parking revenue chart is provided as part of the analytics interface.

The current video session revenue is displayed dynamically from the Flask-generated statistics.

### 🖼️ ROI Detection Grid

The system saves processed parking-region images as ROI snapshots.

These snapshots can be viewed in the analytics interface to inspect the detected parking regions.

### 📋 Parking Event Log

The analytics page provides an event table containing information such as:

* Slot
* Event
* Time
* Duration
* Billing

The table is populated from the parking-processing logs.

---

# 🧠 Model Performance

The application contains a dedicated **Model Performance** section.

The UI presents a confusion matrix containing the current displayed evaluation values:

| Metric         | Value |
| -------------- | ----: |
| True Positive  |   608 |
| False Negative |     0 |
| False Positive |     8 |
| True Negative  |   601 |

The model page also displays the configured training information:

| Parameter          |              Value |
| ------------------ | -----------------: |
| Epochs             |                  5 |
| Batch Size         |                 32 |
| Optimizer          |               Adam |
| Learning Rate      |               1e-3 |
| Training Samples   |              4,872 |
| Validation Samples |              1,218 |
| Training Time      | ~12 minutes on CPU |

These values are presented by the project's model-performance UI; they should be treated as the project's displayed experiment information rather than independently reproduced benchmark results.

---

# 🛠️ Technology Stack

| Technology             | Purpose                              |
| ---------------------- | ------------------------------------ |
| **Python**             | Backend and AI processing            |
| **Flask**              | Web application framework            |
| **OpenCV**             | Video/image processing               |
| **TensorFlow / Keras** | CNN model inference                  |
| **NumPy**              | Numerical and image-array processing |
| **HTML5**              | Web interface                        |
| **CSS**                | UI styling                           |
| **JavaScript**         | Frontend interactions and charts     |
| **Bootstrap 5**        | Responsive UI                        |
| **Chart.js**           | Analytics visualization              |
| **Font Awesome**       | UI icons                             |

The project itself identifies Python, Flask, OpenCV, TensorFlow CNN, NumPy, JavaScript, and Bootstrap as its technology stack.

---

# 📁 Project Structure

```text
Smart-Ai-Vision-Based-Parking-System/
│
├── app.py
├── app-checkpoint.py
├── test_model.py
│
├── base.html
├── index.html
├── dashboard.html
├── model.html
├── analytics.html
├── about.html
│
├── README.md
│
└── [Required model / static assets]
```

### Main Files

#### `app.py`

The main Flask application.

Responsibilities include:

* Flask configuration
* Video uploading
* Video processing
* OpenCV processing
* CNN model loading
* Parking classification
* Parking-state tracking
* Entry/exit logging
* Parking-duration calculation
* Revenue calculation
* Dashboard statistics
* ROI image generation

#### `app-checkpoint.py`

A checkpoint/previous version of the Flask application.

#### `test_model.py`

A small utility used to load the trained `cnn_model.h5` model and display its TensorFlow/Keras summary.

#### `base.html`

Provides the common application layout and navigation.

The interface contains:

* Dashboard
* Model Performance
* Parking Analytics
* About Project

It also loads Bootstrap, Chart.js, and the application's JavaScript chart logic.

#### `index.html`

The main Jinja template that extends `base.html` and includes:

* Dashboard
* Model Performance
* Analytics
* About

#### `dashboard.html`

Contains the parking upload interface, video preview, parking metrics, and dashboard UI.

#### `model.html`

Contains model-performance information, confusion matrix, and training configuration.

#### `analytics.html`

Contains revenue visualization, ROI detection grid, and parking event logs.

#### `about.html`

Documents the project concept and technology stack.

---

# 💻 Installation

## 1. Clone the Repository

```bash
git clone https://github.com/charishma265/Smart-Ai-Vision-Based-Parking-System.git
```

```bash
cd Smart-Ai-Vision-Based-Parking-System
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
```

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
```

```bash
source venv/bin/activate
```

---

## 3. Install Dependencies

Install the required Python packages:

```bash
pip install flask
pip install opencv-python
pip install numpy
pip install tensorflow
```

Or create a `requirements.txt` containing:

```text
Flask
opencv-python
numpy
tensorflow
Werkzeug
```

Then run:

```bash
pip install -r requirements.txt
```

---

# 🧠 CNN Model Setup

The application requires the trained model:

```text
cnn_model.h5
```

The current `app.py` loads the model using a Windows-specific absolute path:

```text
C:\Users\ADMIN\Downloads\realtimeparking\realtimeparking\cnn_model.h5
```

Therefore, before running the application on another computer, update the model path to point to the location of your own `cnn_model.h5`.

### Recommended Project Structure

For easier deployment, place the model inside the repository:

```text
Smart-Ai-Vision-Based-Parking-System/
│
├── model/
│   └── cnn_model.h5
│
├── app.py
├── test_model.py
└── ...
```

Then load it using a relative path instead of a machine-specific path.

---

# ▶️ Running the Application

Start the Flask application:

```bash
python app.py
```

The current application is configured to run on:

```text
http://127.0.0.1:5001
```

The Flask entry point uses port `5001`.

Open your browser and visit:

```text
http://127.0.0.1:5001
```

---

# 🔄 Application Workflow

```text
1. Start Flask Server
        ↓
2. Open SmartParkAI Dashboard
        ↓
3. Upload Parking/CCTV Video
        ↓
4. Preview Uploaded Video
        ↓
5. Process Video
        ↓
6. OpenCV Extracts Frames
        ↓
7. Parking Regions Are Generated
        ↓
8. CNN Classifies Each Region
        ↓
9. Parking State Is Updated
        ↓
10. Entry/Exit Events Are Logged
        ↓
11. Parking Duration Is Calculated
        ↓
12. Revenue Is Calculated
        ↓
13. Dashboard & Analytics Updated
```

---

# 📌 Key Functional Modules

### 🎥 Video Processing

Processes uploaded parking videos using OpenCV.

### 🧠 AI Classification

Uses a TensorFlow/Keras CNN model to classify parking regions.

### 🅿️ Occupancy Detection

Determines whether a parking region is occupied or empty.

### ⏱️ Duration Tracking

Tracks how long a vehicle remains in a parking region.

### 💰 Revenue Calculation

Calculates parking charges from recorded parking duration.

### 📊 Analytics

Displays parking occupancy, revenue, ROI snapshots, and event information.

### 🌐 Web Dashboard

Provides a centralized interface for operating and monitoring the system.

---

# 🎓 Use Cases

SmartParkAI can be adapted for:

* 🏫 College and university parking
* 🏢 Corporate office parking
* 🏬 Shopping malls
* 🏥 Hospitals
* 🏨 Hotels
* 🚉 Railway and transit parking
* ✈️ Airport parking
* 🏙️ Smart-city parking
* 🏘️ Residential communities

---

# 🚀 Future Enhancements

The current implementation can be extended into a more production-ready smart parking platform.

### AI Improvements

* Improve CNN accuracy
* Add data augmentation
* Support different camera angles
* Handle night-time conditions
* Improve performance under rain and shadows
* Replace grid-based ROI detection with automatic parking-slot detection
* Add YOLO-based vehicle detection

### Application Improvements

* Real-time CCTV streaming
* Multiple camera support
* User authentication
* Admin dashboard
* Parking reservation
* Vehicle/license-plate recognition
* Automated notifications
* Cloud deployment
* Database integration
* Historical parking analytics

### Billing Improvements

* Hourly parking rates
* Different vehicle categories
* Dynamic pricing
* Digital payment integration
* Parking receipts

---

# ⚠️ Current Limitations

The current version is a project/prototype implementation rather than a fully production-ready parking platform.

Important considerations:

* Parking regions are generated using a configured grid rather than automatically detecting arbitrary parking spaces.
* The CNN model must be available locally.
* The current model path in `app.py` is machine-specific.
* Billing uses a demonstration rate of ₹5 per 4 seconds.
* Video processing is performed on uploaded footage.
* The repository does not currently include a `requirements.txt` file or the CNN model file in the visible repository file list.

---

# 🔐 Security & Deployment Notes

Before deploying publicly:

* Do not commit private credentials.
* Do not commit unnecessary large video files.
* Store model files appropriately.
* Replace hard-coded local paths.
* Add file-type validation.
* Add upload-size limits.
* Use production Flask deployment such as Gunicorn on Linux.
* Add authentication for administrative dashboards.
* Store persistent parking data in a database.

---

# 📚 Learning Outcomes

This project demonstrates practical implementation of:

* Python programming
* Flask web development
* Computer Vision
* OpenCV
* Deep Learning
* Convolutional Neural Networks
* TensorFlow/Keras
* Image preprocessing
* Video processing
* Region of Interest extraction
* Classification
* Event/state tracking
* Data visualization
* Web dashboard development

---

# 👩‍💻 Contributors

**Project:** SmartParkAI – Vision-Based Smart Parking Detection System

**Repository:** `charishma265/Smart-Ai-Vision-Based-Parking-System`

GitHub:

https://github.com/charishma265/Smart-Ai-Vision-Based-Parking-System

---

# ⭐ Support

If you find this project useful, consider giving the repository a ⭐ on GitHub.

Contributions, suggestions, and improvements are welcome.

---

# 📄 License

No explicit license is currently provided in the repository.

If you plan to make the project open source, consider adding an appropriate license such as the MIT License.

---

## 🚗 SmartParkAI

**See the parking lot. Understand the occupancy. Track the activity. Make parking smarter.**
