from keras.models import load_model
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.utils import img_to_array
# from keras.preprocessing.image import img_to_array
from keras.preprocessing import image

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
model = tf.keras.models.load_model('emo_model.h5')

emo_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Suprise']

cap = cv2.VideoCapture(0)


while True:
    ret, frame = cap.read()
    labels = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        roi_gray = gray[y:y+h , x:x+w]
        roi_gray = cv2.resize(roi_gray, (224,224))
        # roi_gray = np.expand_dims(roi_gray, axis=0)
        # final_image = cv2.resize(frame, (224, 224))

        # final_image = cv2.resize(frame, (224, 224))
        # final_image = np.expand_dims(final_image, axis=0)
        # final_image = final_image / 255.0




        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)



            prediction = model.predict(roi)[0]
            label = emo_labels[prediction.argmax()]
            label_position = (x,y-10)
            cv2.putText(frame, label, label_position,cv2.FONT_ITALIC, 1, (225,0,0),2)
        else:
            cv2.putText(frame, 'NO Faces', (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,225), 2)

    cv2.imshow("Face Emotion Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


