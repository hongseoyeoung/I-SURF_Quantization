from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input, decode_predictions
from keras.models import Model
import numpy as np
# import to use quantify.py
from check import check

# # vgg19 model
model = VGG19(weights='imagenet')

# transform image into input 
img_path = 'mug.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# check specific mask array
# quanti_array = ['fix8','fix8','fl16','fix8','fl16','fix8'
#                 ,'fix8','fl16','fl16','fl16','fl16','fl16'
#                   ,'fl16','fl16','fl16','fl16','fl16','fl16','fl16']
# quanti_array = ['original']*19
# quanti_array = ['fix8']*19
quanti_array = ['prun_0.003']*19
# quanti_array = ['bitwiseAnd_21','bitwiseAnd_16','bitwiseAnd_17','bitwiseAnd_22','bitwiseAnd_17','bitwiseAnd_22'
#                 ,'bitwiseAnd_17','bitwiseAnd_22','bitwiseAnd_17','bitwiseAnd_17','bitwiseAnd_17','bitwiseAnd_21'
#                   ,'bitwiseAnd_17','bitwiseAnd_17','bitwiseAnd_17','bitwiseAnd_17','bitwiseAnd_17','bitwiseAnd_17','bitwiseAnd_17']     
#             
check(model, quanti_array, x)