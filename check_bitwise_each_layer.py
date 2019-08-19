from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input, decode_predictions
from keras.models import Model
import numpy as np

from check import check_each_layer

# # vgg19 model
model = VGG19(weights='imagenet')

# transform image into input 
img_path = 'mug.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# only one layer applying mask
mask_array = ['bitwiseAnd_1', 'bitwiseAnd_2', 'bitwiseAnd_4', 'bitwiseAnd_8', 
                'bitwiseAnd_16', 'bitwiseAnd_17', 'bitwiseAnd_18', 'bitwiseAnd_19', 
                    'bitwiseAnd_20', 'bitwiseAnd_21', 'bitwiseAnd_22']
check_each_layer(model, mask_array, x)
