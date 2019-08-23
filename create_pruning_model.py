from keras.applications.vgg19 import VGG19
from keras.models import Model
from convert import pruning

# # vgg19 model
model = VGG19(weights='imagenet')

pruning(model, 0.005)