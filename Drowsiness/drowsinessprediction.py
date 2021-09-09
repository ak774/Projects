
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image,ImageOps

import numpy as np
import cv2
loaded_model = load_model('drowsiness.h5')
loaded_model.layers[0].input_shape

#image_path='C:\\Users\\akshay goel\\OneDrive\\Documents\\Drowsiness_project\\116.jpg'
#img = image.load_img(image_path,color_mode="grayscale", target_size=(24, 24))
video = cv2.VideoCapture(0)

while True:
  _, frame = video.read()
  im=Image.fromarray(frame)
  im=ImageOps.grayscale(im)
  im = im.resize((24,24))
  img = np.array(im)
  img = np.expand_dims(img, axis=0)
  img=img.reshape(-1, 24, 24, 1)
  predict_x=loaded_model.predict(img)
  print(predict_x)
  classes_x=np.argmax(predict_x,axis=1)
  if classes_x==0:
    classes='Closed'
  elif classes_x==1:
    classes='Open'
  elif classes_x==2:
    classes='No Yawn'
  elif classes_x==3:
    classes='Yawn'
  height,width = frame.shape[:2]
  cv2.rectangle(frame, (0,height-50) , (200,height) , (255,0,0) , thickness=cv2.FILLED )
  cv2.putText(frame,classes,(100,height-20), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1,cv2.LINE_AA)
  cv2.imshow("prediction",frame)
  cv2.waitKey(0)
video.release()
cv2.destroyAllWindows()
  




