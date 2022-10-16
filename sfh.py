

import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import numpy as np

model = tf.keras.models.load_model('emo_model.h5')
#
# faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#
# def detect(gray, frame):
#     faces = faceCascade.detectMultiScale(gray,1.3,5)
#     for (x,y,w,h) in faces:
#         cv2.rectangle(frame, (x,y), (x+w , y+h), (255,0,0),2)
#         roi_gray = gray[y:y+h, x:x+w]
#         roi_color = frame[y:y+h, x:x+w]
#         face = faceCascade.detectMultiScale(roi_gray,1.1,3)
#         for (ex, ey, ew, eh) in face:
#
#             face_roi = roi_color[ey: ey + eh, ex:ex + ew]
#     return frame
#
#
# cap = cv2.VideoCapture(0)
#
# while True:
#     ret, frame = cap.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     canvas = detect(gray, frame)
#     cv2.imshow("video", canvas)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()



# cap = cv2.VideoCapture(0)
#
# # if not cap.isOpened():
# #   cap = cv2.VideoCapture(0)
# # if not cap.isOpened():
# #   raise IOError("Cannot Open Webcam")
#
# while cap.isOpened():
#
#     ret, frame = cap.read()
#
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     faces = faceCascade.detectMultiScale(gray, 1.1, 4)
#
#     for x, y, w, h in faces:
#
#         roi_gray = gray[y:y + h, x:x + w]
#
#         roi_color = frame[y:y + h, x:x + w]
#
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (225, 0, 0), 2)
#
#         face = faceCascade.detectMultiScale(roi_gray, 1.1, 3)
#
#         for (ex, ey, eh, ew) in face:
#             face_roi = roi_color[ey: ey + eh, ex:ex + ew]


font_scale = 1.5
font = cv2.FONT_ITALIC

rectangle_bgr = (225, 225, 225)
img = np.zeros((500, 500))

text = 'some text in a box!'

(text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]

text_offset_x = 10
text_offset_y = img.shape[0] - 25

box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height - 2))
cv2.rectangle(img, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
cv2.putText(img, text, (text_offset_x, text_offset_y), font, fontScale=font_scale, color=(0, 0, 0), thickness=1)

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def detect(gray, frame):
    faces = faceCascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        face = faceCascade.detectMultiScale(roi_gray, 1.1, 3)
        for (ex, ey, ew, eh) in face:
            cv2.rectangle(roi_color, (ex, ey),(ex+ew , ey+eh), (0,255,0),2)
    return frame


cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas = detect(gray, frame)


    final_image = cv2.resize(frame, (224, 224))
    final_image = np.expand_dims(final_image, axis=0)
    final_image = final_image / 255.0

    font = cv2.FONT_HERSHEY_SIMPLEX

    prediction = model.predict(final_image)

    font_scale = 1.5
    font = cv2.FONT_ITALIC

    if (np.argmax(prediction) == 0):
        status = 'Angry'

        x1, y1, w1, h1 = 0, 0, 175, 75
        cv2.rectangle(frame, (x1, x1), (x1 + w1, y1 + h1), (0, 0, 0), -1)
        cv2.putText(frame, status, (x1 + int(w1 / 10), y1 + int(h1 / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 225), 2)


        # cv2.putText(frame, status(int(100, 150)), font, 3, (0, 0, 225), 2, cv2.LINE_4)
        #cv2.putText(frame, cv2.FONT_ITALIC, 3, (225, 0, 0), 2,cv2.LINE_4)
        cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), (0, 0, 225))



    elif (np.argmax(prediction) == 1):
        status = 'Disgust'

        x1, y1, w1, h1 = 0, 0, 175, 75
        cv2.rectangle(frame, (x1, x1), (x1 + w1, y1 + h1), (0, 0, 0), -1)
        cv2.putText(frame, status, (x1 + int(w1 / 10), y1 + int(h1 / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 225), 2)

        # cv2.putText(frame, status(100, 150), font, 3, (0, 0, 225), 2, cv2.LINE_4)
        #cv2.putText(frame, cv2.FONT_ITALIC, 3, (225, 0, 0), 2,cv2.LINE_4)
        cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), (0, 0, 225))


    elif (np.argmax(prediction) == 2):
        status = 'Fear'

        x1, y1, w1, h1 = 0, 0, 175, 75
        cv2.rectangle(frame, (x1, x1), (x1 + w1, y1 + h1), (0, 0, 0), -1)
        cv2.putText(frame, status, (x1 + int(w1 / 10), y1 + int(h1 / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 225), 2)

        # cv2.putText(frame, status(100, 150), font, 3, (0, 0, 225), 2, cv2.LINE_4)
        #cv2.putText(frame,  cv2.FONT_ITALIC, 1, (225, 0, 0), 2, cv2.LINE_4)
        cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), (0, 0, 225))

    elif (np.argmax(prediction) == 3):
        status = 'Happy'

        x1, y1, w1, h1 = 0, 0, 175, 75
        cv2.rectangle(frame, (x1, x1), (x1 + w1, y1 + h1), (0, 0, 0), -1)
        cv2.putText(frame, status, (x1 + int(w1 / 10), y1 + int(h1 / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 225), 2)

        # cv2.putText(frame, status(100, 150), font, 3, (0, 0, 225), 2, cv2.LINE_4)
        #cv2.putText(frame,  cv2.FONT_ITALIC, 3, (225, 0, 0), 2,cv2.LINE_4)
        cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), (0, 0, 225))


    elif (np.argmax(prediction) == 4):
        status = 'Neutral'

        x1, y1, w1, h1 = 0, 0, 175, 75
        cv2.rectangle(frame, (x1, x1), (x1 + w1, y1 + h1), (0, 0, 0), -1)
        cv2.putText(frame, status, (x1 + int(w1 / 10), y1 + int(h1 / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 225), 2)

        # cv2.putText(frame, status(100, 150), font, 3, (0, 0, 225), 2, cv2.LINE_4)
        #cv2.putText(frame, cv2.FONT_ITALIC, 3, (225, 0, 0), 2, cv2.LINE_4)
        cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), (0, 0, 225))


    elif (np.argmax(prediction) == 5):
        status = 'Sad'

        x1, y1, w1, h1 = 0, 0, 175, 75
        cv2.rectangle(frame, (x1, x1), (x1 + w1, y1 + h1), (0, 0, 0), -1)
        cv2.putText(frame, status, (x1 + int(w1 / 10), y1 + int(h1 / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 225), 2)

        # cv2.putText(frame, status(100, 150), font, 3, (0, 0, 225), 2, cv2.LINE_4)
        #cv2.putText(frame, status, cv2.FONT_ITALIC, 3, (225, 0, 0), 2, cv2.LINE_4)
        cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), (0, 0, 225))
    else:

        status = 'Suprise'


        x1, y1, w1, h1 = 0, 0, 175, 75
        cv2.rectangle(frame, (x1, x1), (x1 + w1, y1 + h1), (0, 0, 0), -1)
        cv2.putText(frame, status, (x1 + int(w1 / 10), y1 + int(h1 / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 225), 2)

        # cv2.putText(frame, status(100, 150), font, 3, (0, 0, 225), 2, cv2.LINE_4)
        #cv2.putText(frame, cv2.FONT_ITALIC, 3, (225, 0, 0), 2, cv2.LINE_4)
        cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), (0, 0, 225))

    cv2.imshow("Face Emotion Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()




