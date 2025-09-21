import streamlit as st
import requests
from PIL import Image
import io

st.title('🚦 GTSRB Traffic Sign Classifier')
st.write("Upload an image of a traffic Sign to get its predicted label and confidence score")
uploaded_file=st.file_uploader("Upload an image",type=["jpg","jpeg","png"])
if uploaded_file is not None:
    image=Image.open(uploaded_file)
    st.image(image,caption='Uploaded Image',width=250)

    if st.button('Predict'):
        files={'file':(uploaded_file.name,uploaded_file.getvalue(),"image/jpeg")}
        try:
            response = requests.post('http://127.0.0.1:8000/predict', files=files)
            if response.status_code == 200:
                result=response.json()
                st.success(f"**Predicted Label:** {result['predicted_label']}")
                st.info(f"Confidence: {result['confidence_score'] * 100:.2f}%")
            else:
                st.error(f"Server returned status code {response.status_code}")
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the FastAPI server. Make sure it is running.")
