# 🚦 German Traffic Sign Recognition (GTSRB)

##  About the Project
This project detects and classifies German traffic signs using deep learning.  
It demonstrates a complete **Machine Learning pipeline**:
- Data loading & augmentation
- Model training and evaluation
- FastAPI & Streamlit deployment
- Docker containerization for reproducible builds


## 📂 Dataset Details
- **Name:** [German Traffic Sign Recognition Benchmark (GTSRB)](https://benchmark.ini.rub.de/?section=gtsrb&subsection=news)
- **Size:** ~50,000 images of 43 different traffic sign classes
- **Data Split:** Train / Validation / Test
- **Source:** The dataset was imported directly from Kaggle in this project.

---

##  Model Details

- **Base Architecture:** Transfer Learning with **MobileNetV2** (ImageNet weights)
- **Input Size:** 128 × 128 RGB images
- **Fine-tuning:** Last layers unfrozen for better accuracy
- **Performance:** ~**97% validation accuracy** after initial fine-tuning (you can improve by training longer)

## Deployment Details
The project is deployed using :
- Frontend: Streamlit
- Backend: FastAPI,Docker
