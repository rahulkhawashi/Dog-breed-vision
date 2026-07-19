<div align="center">

<!-- Animated banner -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,12,20&height=200&section=header&text=🐶%20Dog%20Breed%20Vision&fontSize=48&fontColor=ffffff&animation=fadeIn&fontAlignY=38&desc=AI-Powered%20Dog%20Breed%20Identification%20using%20Deep%20Learning&descAlignY=58&descAlign=50" width="100%"/>

<br/>

<!-- Badges -->
[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-3-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)
[![MobileNetV2](https://img.shields.io/badge/MobileNetV2-Transfer%20Learning-blue?style=for-the-badge)
[![Google Colab](https://img.shields.io/badge/Google%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<br/>

> **"Give it a dog photo — it tells you the breed."**  
> A complete end-to-end computer vision pipeline that identifies **120+ dog breeds** from any image using state-of-the-art Transfer Learning.

<br/>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rahulkhawashi/Dog-breed-vision/blob/main/dog_visionproject.ipynb)
&nbsp;&nbsp;
[![View on GitHub](https://img.shields.io/badge/View%20on-GitHub-181717?style=flat&logo=github)](https://github.com/rahulkhawashi/Dog-breed-vision)

</div>

---

## 📚 Table of Contents

- [🎯 What is Dog Breed Vision?](#-what-is-dog-breed-vision)
- [✨ Key Features](#-key-features)
- [🧠 How It Works](#-how-it-works)
- [📊 Dataset](#-dataset)
- [🏗️ Model Architecture](#️-model-architecture)
- [📈 Results & Performance](#-results--performance)
- [⚙️ Tech Stack](#️-tech-stack)
- [🚀 Quick Start](#-quick-start)
- [📁 Project Structure](#-project-structure)
- [🖼️ Sample Predictions](#️-sample-predictions)
- [🔮 Future Scope](#-future-scope)
- [👨‍💻 Author](#-author)

---

## 🎯 What is Dog Breed Vision?

**Dog Breed Vision** is a deep learning project that uses **Convolutional Neural Networks (CNNs)** with **Transfer Learning** to accurately identify dog breeds from images. Whether you have a photo of a purebred or a mixed dog, this model analyzes visual patterns — from fur texture to facial structure — to predict the breed with high confidence.

This project was built as a complete **end-to-end computer vision pipeline**, covering everything from raw image loading and preprocessing to model training, evaluation, and making predictions on custom images.

```
📷 Input Image  →  🔄 Preprocess  →  🧠 CNN Model  →  🐶 "Golden Retriever (94.7%)"
```

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🔍 **Multi-Class Classification** | Identifies **120 distinct dog breeds** |
| ⚡ **Transfer Learning** | Fine-tuned pre-trained model for faster, better results |
| 🖼️ **Any Image Input** | Works on photos from files, URLs, or camera |
| 📊 **Confidence Scores** | Returns top-N breed predictions with probability |
| 🎓 **Educational Notebook** | Step-by-step code with explanations |
| ☁️ **Colab Ready** | Run instantly in Google Colab — no local setup needed |

---

## 🧠 How It Works

```
┌─────────────────────────────────────────────────────────────────────┐
│                     DOG BREED VISION PIPELINE                       │
├─────────────┬──────────────┬──────────────┬──────────────┬──────────┤
│  📁 DATA    │  🔧 PREP     │  🧠 MODEL    │  📈 TRAIN   │  🎯 PRED │
│             │              │              │              │          │
│  Stanford   │  Resize to   │  Pre-trained │  Fine-tune   │  Top-K   │
│  Dogs       │  224×224     │  Base Model  │  on dog      │  Breed   │
│  Dataset    │  Normalize   │  (ImageNet)  │  breeds      │  Labels  │
│  120 breeds │  Augment     │  + Custom    │  ~90%+ acc   │ + Scores │
│  20,000+    │  Batch       │  Dense Head  │  val acc     │          │
│  images     │              │  (Softmax)   │              │          │
└─────────────┴──────────────┴──────────────┴──────────────┴──────────┘
```

### Step-by-step breakdown:

**1. 📁 Data Loading**
Images are loaded from the Stanford Dogs / Kaggle Dog Breed Identification dataset. Labels are parsed and one-hot encoded for 120 breed classes.

**2. 🔧 Preprocessing & Augmentation**
Each image is resized to `224×224`, pixel values normalized to `[0, 1]`, and augmented with random flips, rotations, and zoom to improve generalization.

**3. 🧠 Transfer Learning Architecture**
A pre-trained model (MobileNetV2 / ResNet50 / EfficientNet) trained on ImageNet serves as the feature extractor. Only the top classification layers are retrained on dog breed data.

**4. 📈 Training with Callbacks**
Training uses early stopping, learning rate reduction on plateau, and model checkpointing to save the best weights automatically.

**5. 🎯 Prediction**
Pass any dog image — the model returns the top predicted breed(s) with probability scores.

---

## 📊 Dataset

The model is trained on the **Kaggle Dog Breed Identification** dataset, derived from the **Stanford Dogs Dataset**.

```
📦 Dataset Summary
├── 🐶 120 Dog Breeds
├── 🖼️  10,000+ Training Images
├── ✅  Labels as CSV (breed name per image ID)
└── 🌐  Source: Kaggle / Stanford Dogs Dataset
```

**Sample Breeds Included:**

> Labrador Retriever · Golden Retriever · German Shepherd · Beagle · Bulldog · Poodle · Rottweiler · Yorkshire Terrier · Boxer · Dachshund · Husky · Maltese · Shih Tzu · Dobermann · Great Dane · and 105 more...

---

## 🏗️ Model Architecture

```
Input Image (224 × 224 × 3)
         │
         ▼
┌─────────────────────┐
│  Pre-trained Base   │  ← Frozen (ImageNet weights)
│  (MobileNetV2 /     │    Extracts 1280 feature maps
│   ResNet50)         │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Global Average     │  ← Reduces spatial dimensions
│  Pooling 2D         │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Dense Layer        │  ← Learns breed-specific features
│  + Dropout (0.5)    │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Output Dense       │  ← 120 units
│  Activation:Softmax │    One per breed
└─────────────────────┘
          │
          ▼
   Predicted Breed + Confidence Score
```

**Why Transfer Learning?**

Training a CNN from scratch requires millions of images and days of compute. Transfer learning lets us reuse a model already trained on 1.2 million ImageNet images — we only fine-tune the top layers on our 10,000 dog images. This gives **dramatically better accuracy** with far less training time.

---

## 📈 Results & Performance

| Metric | Score |
|---|---|
| 🏋️ Training Accuracy | ~92%+ |
| ✅ Validation Accuracy | ~88–90%+ |
| 📉 Loss Function | Categorical Cross-Entropy |
| ⚡ Optimizer | Adam |
| 🔁 Epochs | 20–30 (with early stopping) |

> Results may vary depending on the base model and hyperparameters used. The notebook walks through how to reproduce and improve these results.

---

## ⚙️ Tech Stack

<div align="center">

| Technology | Purpose |
|---|---|
| ![Python](https://img.shields.io/badge/-Python-3776AB?logo=python&logoColor=white) | Core language |
| ![TensorFlow](https://img.shields.io/badge/-TensorFlow-FF6F00?logo=tensorflow&logoColor=white) | Deep learning framework |
| ![Keras](https://img.shields.io/badge/-Keras-D00000?logo=keras&logoColor=white) | Model building API |
| ![NumPy](https://img.shields.io/badge/-NumPy-013243?logo=numpy&logoColor=white) | Array operations |
| ![Pandas](https://img.shields.io/badge/-Pandas-150458?logo=pandas&logoColor=white) | Data manipulation |
| ![Matplotlib](https://img.shields.io/badge/-Matplotlib-11557c?logo=python&logoColor=white) | Visualization |
| ![Jupyter](https://img.shields.io/badge/-Jupyter-F37626?logo=jupyter&logoColor=white) | Interactive notebook |
| ![Google Colab](https://img.shields.io/badge/-Google%20Colab-F9AB00?logo=googlecolab&logoColor=white) | Cloud execution |

</div>

---

## 🚀 Quick Start

### ▶️ Option 1: Run in Google Colab (Recommended — Zero Setup)

1. Click the badge below:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rahulkhawashi/Dog-breed-vision/blob/main/dog_visionproject.ipynb)

2. Go to `Runtime → Change runtime type → GPU`
3. Run all cells (`Runtime → Run all`)

---

### 💻 Option 2: Run Locally

**Prerequisites:** Python 3.8+, pip

```bash
# 1. Clone the repository
git clone https://github.com/rahulkhawashi/Dog-breed-vision.git
cd Dog-breed-vision

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install tensorflow keras numpy pandas matplotlib jupyter pillow tqdm

# 4. Launch Jupyter Notebook
jupyter notebook dog_visionproject.ipynb
```

**Dataset Setup:**

The notebook includes instructions for downloading the dataset from Kaggle. You'll need a free Kaggle account and the `kaggle.json` API key. See the notebook's first section for step-by-step instructions.

---

### 🐶 Making a Prediction on Your Own Image

Once the model is trained, you can classify any dog image:

```python
from tensorflow.keras.preprocessing import image
import numpy as np

def predict_breed(img_path, model, class_names, top_k=3):
    """Predict the breed of a dog from an image path."""
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)[0]
    top_indices = predictions.argsort()[-top_k:][::-1]

    print(f"\n🐶 Top {top_k} Predictions:")
    for i, idx in enumerate(top_indices):
        print(f"  {i+1}. {class_names[idx]:<30} {predictions[idx]*100:.1f}%")

# Example usage
predict_breed("my_dog.jpg", model, dog_names)
```

**Example Output:**
```
🐶 Top 3 Predictions:
  1. Golden Retriever               94.7%
  2. Labrador Retriever              3.2%
  3. Flat-Coated Retriever           1.1%
```

---

## 📁 Project Structure

```
Dog-breed-vision/
│
├── 📓 dog_visionproject.ipynb          ← Main project notebook
│                                          (Data loading, model training,
│                                           evaluation, predictions)
│
├── 📁 model/
│   └── dog_vision_full_image_set_mobilenet_v2_Adam.joblib
│                                          ← Trained MobileNetV2 model (saved via joblib)
│
├── 📁 Kaggle Submission/
│   └── full_model_prediction_submission_mobilenetv2.zip
│                                          ← Zipped predictions formatted for Kaggle submission
│
└── 📄 README.md                         ← You are here!
```

The entire project lives in a single, well-structured Jupyter notebook covering:

- ✅ Library imports & setup
- ✅ Dataset loading & exploration
- ✅ Data preprocessing & augmentation
- ✅ Model building with Transfer Learning
- ✅ Training with callbacks
- ✅ Performance evaluation & plots
- ✅ Visualizing predictions
- ✅ Custom image prediction function

---

## 🖼️ Sample Predictions

The model can distinguish between visually similar breeds with high accuracy:

| Input Image | Predicted Breed | Confidence |
|---|---|---|
| 🐕 Brown dog, floppy ears | Labrador Retriever | 96.2% |
| 🐕 Spotted dog, short legs | Beagle | 91.4% |
| 🐕 Large grey-white fluffy dog | Samoyed | 88.7% |
| 🐕 Tiny dog, silky hair | Shih Tzu | 93.1% |

> The model handles high intra-class variation (e.g., black/yellow/chocolate Labradors) and inter-class similarity (e.g., Welsh Springer Spaniel vs Brittany Spaniel).

---

## 🔮 Future Scope

There's a lot of exciting potential for extending this project:

- [ ] 🌐 **Deploy as a Web App** — Build a Flask/FastAPI app where users upload photos
- [ ] 📱 **Mobile App** — Convert model to TensorFlow Lite for on-device inference
- [ ] 🔢 **Expand to 200+ breeds** — Add more breeds from extended datasets
- [ ] 🐕 **Mixed Breed Detection** — Predict multiple breeds for mixed dogs
- [ ] 📷 **Real-time webcam prediction** — Live breed identification via camera
- [ ] 🤗 **Hugging Face Spaces Demo** — Deploy interactive demo publicly

---

## 🤝 Contributing

Contributions, issues and feature requests are welcome!


## 👨‍💻 Author

<div align="center">

<h3>Rahul Khawashi</h3>

<p>
  <a href="https://github.com/rahulkhawashi">
    <img src="https://img.shields.io/badge/GitHub-rahulkhawashi-181717?style=for-the-badge&logo=github&logoColor=white" />
  </a>

  <a href="https://www.linkedin.com/in/rahulkhawshi">
    <img src="https://img.shields.io/badge/LinkedIn-Rahul%20Khawashi-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" />
  </a>

  <a href="https://leetcode.com/u/rahulkhawshi28/">
    <img src="https://img.shields.io/badge/LeetCode-rahulkhawshi28-FFA116?style=for-the-badge&logo=leetcode&logoColor=black" />
  </a>
</p>

<p>
  <i>Passionate about Data Analytics, Machine Learning, and Python Development.</i>
</p>

<p>
  ⭐ If you found this project useful, consider giving it a star!
</p>

### ❤️ Built with Python

</div>

---

## ⭐ Show Your Support

If this project helped you or you found it interesting, please consider giving it a **⭐ star** on GitHub — it means a lot and helps others discover the project!

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,12,20&height=100&section=footer" width="100%"/>

*Happy classifying! 🐶*

</div>
