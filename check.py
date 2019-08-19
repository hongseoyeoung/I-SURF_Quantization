import numpy as np
from keras.applications.vgg19 import preprocess_input, decode_predictions
from swap_layer import swap_layer

def check_each_layer(model, mask_array, x):
    origin_weights = model.get_weights()
    num_of_layer = len(origin_weights) // 2 
    
    for i in range(num_of_layer):
        checkfile = open('check_each_layer.txt', 'a')
        for j in mask_array:
            check_model = swap_layer(model, i, j)
            print("{}, {}complete".format(i, j))
            yhat = check_model.predict(x)
            # convert the probabilities to class labels
            label = decode_predictions(yhat)
            # print the classification
            checkfile.write('%s layer, %s bit : %s (%.2f%%) %s (%.2f%%) %s (%.2f%%)\n'
                 % (i, j, label[0][0][1], label[0][0][2]*100, label[0][1][1], label[0][1][2]*100, label[0][2][1], label[0][2][2]*100))
            model.set_weights(origin_weights)
        checkfile.write('\n')
        checkfile.close()

def check(model, quanti_layer, x):
    j = 0
    for i in quanti_layer:
        checkfile = open('check_sens.txt', 'a')
        check_model = swap_layer(model, j, i)
        print("{}, {}complete".format(j, i))
        yhat = check_model.predict(x)
        # convert the probabilities to class labels
        label = decode_predictions(yhat)
        # print the classification
        checkfile.write('0~%s layer, %s bit : %s (%.2f%%) %s (%.2f%%) %s (%.2f%%)\n' 
                % (j, i, label[0][0][1], label[0][0][2]*100, label[0][1][1], label[0][1][2]*100, label[0][2][1], label[0][2][2]*100))
        # model.set_weights(weights)
        checkfile.close()
        j += 1

