from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input, decode_predictions
from keras.models import Model
from keras.models import model_from_json
import numpy as np

# # vgg19 model
model = VGG19(weights='imagenet')

# transform image into input 
img_path = 'mug.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# check accuracy by json, h5
for i in (1, 2, 4, 8, 16, 17, 18, 19, 20, 21, 22):
    testfile = open('test.txt', 'a')
    # load json and create model
    json_file = open('./vgg19/bitwiseAnd_{}/model.json'.format(i), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    quantified_model = model_from_json(loaded_model_json)
    # load weights into new model
    quantified_model.load_weights("./vgg19/bitwiseAnd_{}/model.h5".format(i))
    print("Loaded model from disk")

    yhat = quantified_model.predict(x)
    # convert the probabilities to class labels
    label = decode_predictions(yhat)
    # retrieve the most likely result, e.g. highest probability
    
    testfile.write('%s bit : %s (%.2f%%) %s (%.2f%%) %s (%.2f%%)\n' % (i, label[0][0][1], label[0][0][2]*100, label[0][1][1], label[0][1][2]*100, label[0][2][1], label[0][2][2]*100))
    testfile.close()