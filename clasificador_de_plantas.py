#!/usr/bin/env python
# coding: utf-8

# In[23]:
import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

print(tf.__version__)


# In[24]:
train_folder_path="./data/entrenamiento"
data=[]
img_size=150
for img in os.listdir(train_folder_path):
    img = cv2.imread(os.path.join(train_folder_path,img))
    img_resize= cv2.resize(img,(img_size,img_size))
    data.append(img_resize)
Images = np.array(data)
print(Images.shape)


# In[25]:
etq0=np.repeat(0,13)
etq1=np.repeat(1,13)
etq2=np.repeat(2,13)
etq3=np.repeat(3,13)
etq4=np.repeat(4,13)
etq5=np.repeat(5,13)
etq6=np.repeat(6,13)
etq7=np.repeat(7,13)
etq8=np.repeat(8,13)
etq9=np.repeat(9,13)
etq10=np.repeat(10,13)
etq11=np.repeat(11,13)
etq12=np.repeat(12,13)
etq13=np.repeat(13,13)
etq14=np.repeat(14,13)


# In[26]:
class_names=['Acer Campestre','Acer Capillipes','Acer Cincinatum','Acer Mono','Acer Opalus','Acer Palmatum','Acer Pictum',
                 'Acer Platanoids','Acer Rubrum','Acer Rufinerve','Acer Saccharinum','Alnus Cordata','Alnus Maximowiczii',
                 'Alnus Rubra','Alnus Sieboldiana']


# In[27]:
labels=np.concatenate([etq0,etq1,etq2,etq3,etq4,etq5,etq6,etq7,etq8,etq9,etq10,etq11,etq12,etq13,etq14])
Labels=np.array(labels)
print(Labels.shape)


# In[28]:
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(150, 150,3)),
    keras.layers.Dense(128, activation='relu'),
    
    keras.layers.Dense(15, activation='softmax'),
    
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(Images, Labels, epochs=30)
trained=model.fit(Images, Labels, epochs=50)


# In[44]:
img=cv2.imread("./data/pruebas/Alnus_Sieboldiana_15.ab.jpg")
img_cvt=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.imshow(img_cvt)
plt.show()


# In[46]:
img2=img_cvt
img2=cv2.resize(img2,(img_size,img_size))
#print(img2.shape)
img2=(np.expand_dims(img2,0))
#print(img2.shape)
prediction_single=model.predict(img2)
print("La planta ingresada pertenece a la especie:",class_names[np.argmax(prediction_single)])
