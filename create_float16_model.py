from keras.applications.vgg19 import VGG19
from keras.models import Model
from convert import convert_fl32_to_fl16

# # vgg19 model
model = VGG19(weights='imagenet')

''' convert to float 16bit'''
convert_fl32_to_fl16(model)