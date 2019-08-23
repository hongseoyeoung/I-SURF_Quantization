from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input, decode_predictions
from keras.models import Model
from keras.models import model_from_json
import tensorflow as tf
import numpy as np
# import to use quantify.py
from quantify import quantify
from swap_layer import swap_layer
from check import check, check_each_layer
from model_info import get_layer_name
from convert import *

# # vgg19 model
model = VGG19(weights='imagenet')

print(model.summary())
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


# quantify(model, 1, 2, 4, 8, 16, 17, 18, 19, 20, 21, 22)


# # check accuracy by json, h5
# for i in (1, 2, 4, 8, 16, 17, 18, 19, 20, 21, 22):
#     testfile = open('test.txt', 'a')
#     # load json and create model
#     json_file = open('./vgg19model/model_{}.json'.format(i), 'r')
#     loaded_model_json = json_file.read()
#     json_file.close()
#     quantified_model = model_from_json(loaded_model_json)
#     # load weights into new model
#     quantified_model.load_weights("./vgg19model/model_{}.h5".format(i))
#     print("Loaded model from disk")


#     yhat = quantified_model.predict(x)
#     # convert the probabilities to class labels
#     label = decode_predictions(yhat)
#     # retrieve the most likely result, e.g. highest probability
#     label = label[0][0]
#     # print the classification
    
#     testfile.write('%s layer, %s bit : %s (%.2f%%) %s (%.2f%%) %s (%.2f%%)\n' % (i, j, label[0][0][1], label[0][0][2]*100, label[0][1][1], label[0][1][2]*100, label[0][2][1], label[0][2][2]*100))
#     testfile.close()

# # check each layer mask
# mask_array = [1, 2, 4, 8, 16, 17, 18, 19, 20, 21, 22]
# check_each_layer(model, mask_array, x)


# # check specific mask array
# quanti_array = ['fix8','fix8','fix8','fix8','fl16','fix8'
#                 ,'fl16','fl16','fl16','fl16','fl16','fl16'
#                   ,'fl16','fl16','fl16','fl16','fl16','fl16','fl16']
# check(model, quanti_array, x)

''' convert to float 16bit'''
# convert_fl32_to_fl16(model)

# # prdiect
# yhat = model.predict(x)
# # convert the probabilities to class labels
# label = decode_predictions(yhat)
# # retrieve the most likely result, e.g. highest probability
# label = label[0][0]
# # print the classification
# print('%s (%.2f%%)' % (label[0][0][1], label[0][0][2]*100), '%s (%.2f%%)' % (label[0][1][1], label[0][1][2]*100), '%s (%.2f%%)' % (label[0][2][1], label[0][2][2]*100))


''' convert to fixed 8bit and save weights as file'''
#convert_fl32_to_fix8(model)

''' load and convert fixed 8bit to float32'''
# w = convert_fix8_to_fl32(model)
# model.set_weights(w)

# # prdiect
# yhat = model.predict(x)
# # convert the probabilities to class labels
# label = decode_predictions(yhat)
# # retrieve the most likely result, e.g. highest probability
# label = label[0][0]
# # print the classification
# print('%s (%.2f%%)' % (label[0][0][1], label[0][0][2]*100), '%s (%.2f%%)' % (label[0][1][1], label[0][1][2]*100), '%s (%.2f%%)' % (label[0][2][1], label[0][2][2]*100))


