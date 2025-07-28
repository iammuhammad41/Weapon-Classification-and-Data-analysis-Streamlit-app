
# Weapon Classification & Data Analysis Streamlit App

This repository contains a complete pipeline for:

1. **Training** a ResNet‑18 model via transfer learning on a public weapon‐image dataset.  
2. **Exploratory Data Analysis (EDA)** of your dataset (class counts & sample images).  
3. **Interactive Streamlit App** to upload an image and get real‑time weapon classification.



## 📁 Repository Structure

```

.
├── data/
│   ├── train/                 # training images, organized by class subfolder
│   │   ├── knife/
│   │   ├── pistol/
│   │   └── rifle/
│   └── val/                   # validation images, same structure
│       ├── knife/
│       ├── pistol/
│       └── rifle/
├── models/
│   └── weapon\_resnet18.pth    # saved PyTorch model weights
├── training\_metrics.png       # accuracy/loss curves after training
├── train.py                   # training & evaluation script
├── app.py                     # Streamlit application
└── README.md                  # you are here

````


## 🚀 Quick Start

1. **Install dependencies**  
   ```bash
   pip install torch torchvision streamlit matplotlib seaborn pillow
````

2. **Prepare your dataset**

   * Place labeled weapon images under `data/train/<class>/` and `data/val/<class>/`.
   * Example classes: `knife/`, `pistol/`, `rifle/`, etc.

3. **Train the model**

   ```bash
   python train.py
   ```

   * Trains for 10 epochs by default.
   * Saves weights to `models/weapon_resnet18.pth` and produces `training_metrics.png`.

4. **Run the Streamlit app**

   ```bash
   streamlit run app.py
   ```

   * **Home:** Displays training curves.
   * **EDA:** Shows class distribution & sample images.
   * **Classify:** Upload your own image and see top‑3 predictions.


## 🔧 Configuration

* **Batch size**, **learning rate**, **number of epochs**, etc. can be modified at the top of `train.py`.
* **Model architecture** is ResNet‑18 with a custom final layer matching your number of classes.



## 📊 EDA & Metrics

* `app.py` uses `seaborn` & `matplotlib` to render:

  * Class count bar chart
  * Sample image grid
  * Real‑time prediction probabilities

* `training_metrics.png` shows training vs. validation accuracy & loss curves.



## 🤝 Contributing

Feel free to:

* Add more weapon classes
* Experiment with different architectures
* Improve the UI/UX in Streamlit

