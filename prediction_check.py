from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input, decode_predictions
from keras.models import Model
import numpy as np

# # vgg19 model
model = VGG19(weights='imagenet')

# transform image into input 
img_path = 'mug.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# prdiect
yhat = model.predict(x)
# convert the probabilities to class labels
label = decode_predictions(yhat)

# print the classification
print('%s (%.2f%%)' % (label[0][0][1], label[0][0][2]*100), '%s (%.2f%%)' % (label[0][1][1], label[0][1][2]*100), '%s (%.2f%%)' % (label[0][2][1], label[0][2][2]*100))