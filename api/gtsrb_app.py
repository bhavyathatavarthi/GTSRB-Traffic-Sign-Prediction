from fastapi import FastAPI,UploadFile,File
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import tempfile
from api.map_dict import gtsrb_labels

#Load Model
model=load_model('GTSRB/model/gtsrb_mobilenet.h5')
img_size=(128,128)

app=FastAPI(title='Traffic Sign Classifier')

@app.get("/")
def root():
    return {"message": "GTSRB prediction API is running. Use POST /predict."}
@app.get("/health")
def health_check():
    return{
        'status':'OK',
        'version':'1.0.0',
        'model_loaded':model is not None
    }

@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    tmp=tempfile.NamedTemporaryFile(delete=False)
    tmp.write(await file.read())
    tmp.close()

    #preprocess image
    img=image.load_img(tmp.name, target_size=(128,128))
    x=image.img_to_array(img)/255.0
    x=np.expand_dims(x,axis=0)

    #Make Prediction
    preds=model.predict(x)
    predicted_class=int(np.argmax(preds,axis=1)[0])
    predicted_label=gtsrb_labels[predicted_class]
    confidence = float(np.max(preds))
    return {"predicted_class":predicted_class, "predicted_label":predicted_label,"confidence_score": round(confidence, 3)}
