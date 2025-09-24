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

1. FastAPI Backend
   ```bash
   uvicorn api.gtsrb_app:app --reload
  - API will be available at: http://127.0.0.1:8000/predict
2. Streamlit Backend
    ```bash
    cd streamlit_app
    streamlit run frontend.py
  - UI will be available at: http://localhost:8501
3. Docker
    ```bash
    docker build -t traffic-sign-api .
    docker run -p 8000:8000 traffic-sign-api
## To Reproduce

1. To clone Repository
   ```bash
   git clone https://github.com/bhavyathatavarthi/GTSRB-Traffic-Sign-Prediction.git
   cd traffic-sign-api
2. Create Virtual Environment
   ```bash
   python -m venv venv
   Windwos- venv\Scripts\activate
   macOS/Linux- source venv/bin/activate
3. Install Requirements
   ```bash
   pip install -r requirements.txt
   ```
   

   
   

