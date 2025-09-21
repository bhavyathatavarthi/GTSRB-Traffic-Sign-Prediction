#%%
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
#%%
train_dir = "/kaggle/input/gtsrb-german-traffic-sign/Train"
test_dir  = "/kaggle/input/gtsrb-german-traffic-sign/Test"

img_size   = (128,128)   # for transfer learning
#img_size=(48,48)  #for CNN
batch_size = 32
seed       = 42        # ensure reproducibility of the split

# 2. Split Train -> train/validation

train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=img_size,
    batch_size=batch_size,
    validation_split=0.2,   # 20% goes to validation
    subset="training",
    seed=seed
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=img_size,
    batch_size=batch_size,
    validation_split=0.2,
    subset="validation",
    seed=seed
)
#%%
import tensorflow as tf
from tensorflow.keras import layers

# Random transforms that make sense for traffic signs
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),          
    layers.RandomTranslation(0.05, 0.05),    
    layers.RandomZoom(0.1),               
    layers.RandomContrast(0.1)  
])


normalizer = layers.Rescaling(1./255)
#%%
# Apply augmentation ONLY to training data
train_ds = train_ds.map(lambda x, y: (normalizer(data_augmentation(x, training=True)), y))
val_ds   = val_ds.map(lambda x, y: (normalizer(x), y))

# Improve pipeline performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds   = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
#%%
import os
class_folders = sorted(os.listdir(train_dir))   # all class sub-folders
print("Total number of classes:", len(class_folders))
for cls in sorted(os.listdir(train_dir)):
    n = len(os.listdir(os.path.join(train_dir, cls)))
    print(f"{cls}: {n}")
#%%
class_folders = sorted(os.listdir(train_dir))
counts = []
for cls in class_folders:
    n = len(os.listdir(os.path.join(train_dir, cls)))
    counts.append(n)

print("Counts per class:", counts)
#%%
from sklearn.utils.class_weight import compute_class_weight

classes = np.arange(len(counts)) 

class_weights_arr = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=np.repeat(classes, counts)  
)

# convert to dict for model.fit()
class_weights = dict(zip(classes, class_weights_arr))
print(class_weights)
#%%
num_classes=len(class_folders)
#%%
print(num_classes)
#%%
#BUILDING CNN FROM SCRATCH-METHOD 1
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam

model = Sequential([

    # Conv Block 1
    
    Input(shape=img_size + (3,)),
    Conv2D(32, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.2),


    # Conv Block 2
    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.25),

    # Conv Block 3
    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.3),


    # Flatten + Dense Layers
    Flatten(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(num_classes, activation='softmax')

])
#%%
model.summary()
#%%
from tensorflow.keras.optimizers import Adam
model.compile(optimizer=Adam(1e-3),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
#%%
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau,ModelCheckpoint,CSVLogger
callbacks=[
    EarlyStopping(monitor='val_loss',patience=3,restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss',patience=3,factor=0.2,min_lr=1e-6),
    ModelCheckpoint(filepath='model_check.h5',monitor='val_loss',save_best_only=True,save_weights_only=False,verbose=1),
    CSVLogger('cnn_training_log.csv', append=False)
]
#%%
history= model.fit(train_ds,validation_data=val_ds,epochs=15,callbacks=callbacks)
#%%
plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

#%%
# GTSRB: class-id -> human readable label
gtsrb_labels = {
     0: "Speed limit (20km/h)",
     1: "Speed limit (30km/h)",
     2: "Speed limit (50km/h)",
     3: "Speed limit (60km/h)",
     4: "Speed limit (70km/h)",
     5: "Speed limit (80km/h)",
     6: "End of speed limit (80km/h)",
     7: "Speed limit (100km/h)",
     8: "Speed limit (120km/h)",
     9: "No passing",
    10: "No passing for vehicles over 3.5 metric tons",
    11: "Right-of-way at the next intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Vehicles over 3.5 metric tons prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve to the left",
    20: "Dangerous curve to the right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on the right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End of all speed and passing limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End of no passing by vehicles over 3.5 metric tons"
}

#%%
import tensorflow as tf

tmp_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,        
    image_size=(48,48),  
    batch_size=1         
)
class_names = tmp_ds.class_names
print("Total classes:", len(class_names))
print("Class names:", class_names)

#%%
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array

def predict_image(model, img_path, class_names, img_size=(48,48)):
    
    img = load_img(img_path,target_size=img_size)
    img_array = img_to_array(img) / 255.0     
    img_array = np.expand_dims(img_array, 0)   

    probs = model.predict(img_array)
    class_idx = np.argmax(probs[0])

    predicted_label = class_names[class_idx]
    human_label  = gtsrb_labels[class_idx]
    confidence = probs[0][class_idx]
    plt.imshow(load_img(img_path))
    plt.show()

    return predicted_label,human_label,confidence

#%%
predicted_label,human_label,confidence = predict_image(model,
                            "//kaggle/input/gtsrb-german-traffic-sign/Test/00024.png",
                            class_names)

print(f"Predicted Class: {predicted_label},Label:{human_label},confidence: {confidence:.2%})")
#%%
model.save("/kaggle/working/gtsrb_cnn.h5")
#%%
#BY TRANSFER LEARNING
from tensorflow.keras.applications import ResNet50,EfficientNetB0,MobileNetV2
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
#BY MOBILE NET
base_model=MobileNetV2(weights='imagenet',include_top=False,input_shape=(128,128,3))
base_model.trainable=False
x=GlobalAveragePooling2D()(base_model.output)
x=Dense(256, activation='relu')(x)
x=BatchNormalization()(x)
x=Dropout(0.4)(x)
x=Dense(128,activation='relu')(x)
x=BatchNormalization()(x)
x=Dropout(0.3)(x)
output = Dense(num_classes, activation='softmax')(x)  # 43 classes

model = Model(inputs=base_model.input, outputs=output)

#%%
model.summary()
#%%
model.compile(optimizer=Adam(learning_rate=1e-3),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
#%%
callbacks=[
    EarlyStopping(monitor='val_loss',patience=3,restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss',patience=3,factor=0.2,min_lr=1e-6),
]
#%%
history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=7,
    callbacks=callbacks,
    class_weight=class_weights

)
#%%
callbacks=[
    EarlyStopping(monitor='val_loss',patience=3,restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss',patience=3,factor=0.2,min_lr=1e-6),
    ModelCheckpoint(filepath='mobilenet_model_check.h5',monitor='val_loss',save_best_only=True,save_weights_only=False,verbose=1),
    CSVLogger('mobilenet_training_log.csv', append=False)
]
#%%
#6. Fine-Tune Last Layers
# -------------------------------
for layer in base_model.layers[-40:]:
    layer.trainable = True

model.compile(optimizer=Adam(learning_rate=1e-5),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history_fine1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    class_weight=class_weights,
    callbacks=callbacks
)
#%%
history_fine1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=25,
    initial_epoch=15,
    class_weight=class_weights,
    callbacks=callbacks
)
#%%
plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
plt.plot(history_fine1.history['accuracy'], label='train acc')
plt.plot(history_fine1.history['val_accuracy'], label='val acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.subplot(1,2,2)
plt.plot(history_fine1.history['loss'], label='train loss')
plt.plot(history_fine1.history['val_loss'], label='val loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
#%%
model.save("/kaggle/working/gtsrb_mobilenet.h5")
#%%
