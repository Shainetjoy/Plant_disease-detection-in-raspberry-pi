import pathlib
import cv2
import numpy as np
import os
import tensorflow as tf
import tflite_runtime.interpreter as tflite
from tflite_runtime.interpreter import Interpreter
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.preprocessing import image
import serial,time

model_path = "plant_modelV4.tflite"
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess_frame(img):
   resized_img = cv2.resize(img, (224, 224))  # Resizing the images to be able to pass on MobileNetv2 model
   resized_img = resized_img / 255
   resized_img = resized_img.reshape(1, 224, 224, 3)
   return  resized_img
   print("Preprocessing Function activated......!")

	   
with serial.Serial("/dev/ttyUSB0", 9600, timeout=1) as arduino:
	time.sleep(0.1) #wait for serial to open
	
#time.sleep(0.1) #wait for arduino to answer
	
	video = cv2.VideoCapture(0)
	while True:
	   ret, img = video.read()
	   cv2.imshow('live video', img)
	   

	   input_data = preprocess_frame(img)
	   input_shape = input_details[0]['shape']
	   input_data = np.array(input_data, dtype=np.float32)
	   interpreter.set_tensor(input_details[0]['index'], input_data)
	   interpreter.invoke()
	   output_data = interpreter.get_tensor(output_details[0]['index'])
	   output_data = np.argmax(output_data)
	   print("OUT_PUT......!",output_data)
	   if output_data==0:
		   Output='EARLY_BLIGHT'
		   arduino.write(Output.encode())
		   print("####################>EARLY_BLIGHT")
	   elif output_data ==1:
		   arduino.write(Output.encode())
		   Output='LATE_BLIGHT'
		   print("####################>LATE_BLIGHT")
	   elif output_data ==2:
		   arduino.write(Output.encode())
		   Output='HEALTHY_LEAF'
		   print("####################>HEALTHY_LEAF")
		   
	   else:
		   print("not mach ")

	   if cv2.waitKey(1) & 0xFF == ord('q'):
	       break



	video.release()
	cv2.destroyAllWindows()
