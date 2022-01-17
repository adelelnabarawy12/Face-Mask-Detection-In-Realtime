import cv2 as cv
import numpy as np
from tensorflow.keras.models import load_model

cascPath = 'haarcascade_frontalface_default.xml'
faceCascade = cv.CascadeClassifier(cascPath)
model = load_model('model/bestmodel-93.hdf5')
classes = ['without_mask', 'with_mask']
video_capture = cv.VideoCapture(1)
i = 0
while True:
    ret, frame = video_capture.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=7,
        minSize=(10, 10),
        flags=cv.CASCADE_SCALE_IMAGE
    )
    
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = np.expand_dims(cv.resize(face, (96, 96)), 0)
        temp_pred = model.predict(face) 
        pred = temp_pred.squeeze()
        lbl = int(pred > 0.5)
        color = (0, 255, 0) if lbl == 1 else (0, 0, 255)
        cv.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv.putText(frame, f'{classes[lbl]}: {(pred * 100):.2f}%', (x + 10, y - 10), cv.FONT_HERSHEY_SIMPLEX, .8, (0, 255, 255), 1, cv.LINE_AA)
    # End For
    i += 1
    # cv.imwrite(f'faces/{i}.jpg', frame)
    cv.imshow('camera', frame)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    # End if
# End While

video_capture.release()
cv.destroyAllWindows()