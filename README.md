# SwimAdvisor: High-Stability Pose Estimation for Aquatic Environments 🏊‍♂️

**SwimAdvisor** is a specialized Deep Learning project designed to overcome the unique challenges of human pose estimation in swimming pools. Traditional models often struggle with water refraction, bubbles, and limb occlusion. This project introduces a custom **Biomechanical Smoothing Layer** integrated with fine-tuned **YOLOv8-Pose** models to provide stable, professional-grade movement analysis.

---

## 📂 Project Structure

```text
SwimAdvisor/
├── data/                       # Dataset and test videos
│   └── hadar_check/            # Annotated validation frames (Images/Labels)
├── models/                     # Trained weights (.pt files)
│   ├── level2/                 # Intermediate fine-tuning
│   ├── level3/                 # Advanced training
│   └── level3.5/               # Final optimized model
├── results/                    # Analysis outputs
│   ├── jitter_res/             # CSV reports for temporal stability
│   ├── metrics/                # mAP and static accuracy reports
│   ├── plots/                  # Visualized Jitter & KPI charts
│   └── vids/                   # Processed output videos
├── scripts/                    # Core Logic and Processing
│   ├── smoothing_layer.py      # Stabilization Logic & Biomechanical Guards
│   ├── train.py                # Model training and fine-tuning script
│   ├── benchmarks/             # Performance evaluation
│   │   ├── raw_model_jitter_test.py # Baseline stability testing
│   │   ├── smoothing_impact.py      # Accuracy gain from smoothing layer
│   │   └── static_kpi_test.py       # mAP and static frame evaluation
│   └── data_preprocessing/     # Dataset preparation tools
│       ├── auto_label_and_yaml.py   # Automated labeling pipeline
│       ├── download_datasets.py     # Source data acquisition
│       ├── remove_empty.py          # Data cleaning for generic datasets
│       └── remove_empty_hadar.py    # Specific cleaning for 'Hadar' dataset
├── main.py                     # Primary execution script
└── requirements.txt            # Project dependencies

```

---

## 🛠 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/SwimAdvisor.git
cd SwimAdvisor

```

### 2. Set Up Virtual Environment

```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

```

### 3. Install Dependencies

```bash
pip install -r requirements.txt

```

---

## 🧠 Model Architecture

The project utilizes a **Hybrid Pipeline** approach to ensure data integrity in underwater scenarios:

### I. Fine-Tuned YOLOv8-Pose

We utilized transfer learning on custom swimming datasets, evolving through three levels of training to optimize for underwater keypoint detection (13 points).

### II. The Biomechanical Smoothing Layer

To combat the "Jitter" effect caused by water surface movement, we developed a post-processing layer:

* **Global Anti-Flip:** Prevents left-right limb swapping by analyzing temporal consistency.
* **Anatomical Guards:** Enforces skeletal constraints (e.g., Leg Chain and Torso alignment).
* **Torso-Based Scaling:** Normalizes limb lengths based on torso proportions to handle camera distance variations.
* **Temporal Filtering:** Applies Savitzky-Golay filtering for fluid motion trajectories.

---

## 🚀 Usage / How to Run

### Data Preparation(if you want to train)

Clean and prepare your dataset before training:

```bash
python scripts/data_preprocessing/auto_label_and_yaml.py

```

### Training (optional)

Fine-tune the YOLOv8-Pose model on the swimming dataset:

```bash
python scripts/train.py

```

### Performance Evaluation

Evaluate the stability and accuracy of the results:

```bash
# Evaluate static accuracy (mAP)
python scripts/benchmarks/static_kpi_test.py

# Evaluate the impact of the Smoothing Layer on jitter
python scripts/benchmarks/smoothing_impact.py

```

### Final Inference

Run the full pipeline on a swimming video:

```bash
python main.py

```

---

## 📊 Results

### Temporal Stability (Jitter Reduction)

The **Smoothing Layer** significantly reduces skeletal "shaking" compared to raw model inference:

Keypoint,Raw Jitter (px),Smoothing Layer (px),Improvement (%)
R-Wrist,79.61,16.25,79.59%
R-Elbow,54.54,11.16,79.53%
L-Wrist,69.08,15.66,77.33%
L-Elbow,43.67,10.52,75.91%
Head,30.26,7.48,75.27%
L-Ankle,57.29,19.66,65.69%
L-Shoulder,30.61,10.52,65.64%
R-Ankle,55.14,19.69,64.28%
R-Shoulder,27.53,10.87,60.53%
L-Knee,38.57,15.88,58.81%
R-Knee,36.79,15.94,56.67%
L-Hip,30.46,14.89,51.11%
R-Hip,28.01,14.29,48.97%
AVERAGE,44.74,13.29,66.10%
---

## 📦 Dependencies

* `ultralytics` (YOLOv8)
* `opencv-python`
* `pandas`
* `numpy`
* `scipy`
* `matplotlib`
* `tqdm`

---

## Created by: Ohad Bitton, Hadar Lavsky.

**Final-Year Computer Science Students** *Focus: Computer Vision & Biomechanical Motion Analysis*
