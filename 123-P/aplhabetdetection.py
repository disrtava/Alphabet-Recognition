import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.datasets import fetch_openml 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps 
import os,ssl,time

X = np.load('image.npz')['arr_0']

y = pd.read_csv("labels.csv")["labels"]

print(pd.Series(y).value_counts())

classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

nclasses = len(classes)

x_tr , x_te , y_tr , y_te = train_test_split(X,y,random_state = 9 , train_size = 7500 , test_size = 2500)

x_trscale = x_tr/255.0
x_tescale = x_te/255.0

clf = LogisticRegression(solver = 'saga',multi_class = 'multinomial').fit(x_trscale,y_tr)

y_pred = clf.predict(x_tescale)
accuracy = accuracy_score(y_te , y_pred)
print(accuracy)

cam = cv2.Videocapture(0)

while (True):
    try:
        ret,frame = cam.read()
        
        roi = gray[upper_left[1]:bottom_right[1],upper_left[0]:bottom_right[0]]
        
        height,width = gray.shape
        
        upper_left = (int(width / 2 - 56), int(height / 2 - 56))
        bottom_right = (int(width / 2 + 56), int(height / 2 + 56))
        
        cv2.rectangle(gray,upper_left,bottom_right,(0,56,-0),2)
        
        im_pil = Image.fromarray(roi)
        imagebw = im_pil.convert("L")
        
        imagebewresized = imagebw.resize((28,28),Image.ANTIALIAS)
        
        pixelfilter = 20
        min_pixel = np.percentile(imagebewresized,pixelfilter)
        image_scaled = np.clip(imagebewresized - min_pixel , 0 , 255)
        
        image_scaled = np.asarray(image_scaled)
        
        test_samples = np.array(image_scaled).reshape(1,784)
        test_pred = clf.predict(test_samples)
        print(f'Predicted class is {test_pred}')
        
        cv2.imshow('frame',gray)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        
    except Exception as e:
        pass
        
    cam.release()
    cv2.destroyAllWindows()