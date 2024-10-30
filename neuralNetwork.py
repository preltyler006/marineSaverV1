import os
from imageai.Classification import ImageClassification
import numpy as np
import matplotlib.pyplot as plt
#need to create or download and import data for it to work
#https://keras.io/examples/vision/image_classification_from_scratch/

execution_path = os.getcwd()
prediction = ImageClassification()
prediction.setModelTypeAsResNet50()
prediction.setModelPath(os.path.join(execution_path, "resnet50-19c8e357.pth"))
prediction.loadModel()

predictions, percentage_probabilities = prediction.classifyImage("shutterstock_2263403737-scaled.webp", result_count = 5)
for index in range(len(predictions)):
  print(predictions[index], ": ", percentage_probabilities[index])
