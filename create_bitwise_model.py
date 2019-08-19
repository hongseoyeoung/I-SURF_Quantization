from keras.applications.vgg19 import VGG19
from keras.models import Model
# import to use quantify.py
from quantify import quantify

# # vgg19 model
model = VGG19(weights='imagenet')

quantify(model, 1, 2, 4, 8, 16, 17, 18, 19, 20, 21, 22)