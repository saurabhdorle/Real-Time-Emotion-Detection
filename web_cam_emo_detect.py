
import cv2
import numpy as np
from keras.models import load_model


# Loading pre-trained haar cascade face detect model
cascPath = 'data/haarcascades/haarcascade_frontalface_default.xml' 
faceCascade = cv2.CascadeClassifier(cascPath)
font = cv2.FONT_HERSHEY_SIMPLEX #font type for writing text on screen

# capturing video from webcam
video_capture = cv2.VideoCapture(0)


# Loading save eomtion detection mode
model = load_model('emo_detect.h5')

# Predicting Emotion from Facial expressions using saved model
def get_emotion(img):
    img = img[np.newaxis, :, :] #adding dimension to array
    result = model.predict_classes(img, verbose = 0) #predicting using loaded model
    emotion = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
    return emotion[result[0]]


# Reshaping capture image
def recognize(img):
    img = cv2.resize(img, (48, 48)) #resizing captured image
    img = img.astype('float32')#/255 #changing type to float
    #img = np.asarray(img)
    img = img.reshape(48, 48, 1) #reshaping captured image
    return get_emotion(img)




while True:
    #capturing video frame by frame
    ret, frame = video_capture.read()
    
    flip_frame = cv2.flip(frame, 1) #fliping captured image
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Convert to gray
    
    faces = faceCascade.detectMultiScale(gray, scaleFactor = 1.2, minNeighbors = 5, minSize = (30, 30),
                                         flags = cv2.CASCADE_SCALE_IMAGE) #face detection using haar cascade
    
    
    for (x, y, w, h) in faces:
        
        face = frame[y:y+h, x:x+w, :]
        gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        out = recognize(gray_face) #caling emotion detection function
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2) #drawing rectangle around face
        
        flip_frame = cv2.flip(frame, 1)
        
        cv2.putText(flip_frame, out, (30, 30), font, 1, (255, 255, 0), 2) #writing emotion text on frame
    
    cv2.imshow('rgb', flip_frame)
    
    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break
    
video_capture.release()
cv2.destroyAllWindows()



