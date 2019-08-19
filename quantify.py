import tensorflow as tf
import numpy as np
import os
from model_info import get_layer_name
# import to use bit_wise_And.py
from bit_wise_And import bit_wise_And, masking

# qauntify method has two parameters. 'model' is keras model. 'mask_nums' is a tuple of values to mask. 
def quantify(model, *mask_nums):
    # get the all weights
    weights = model.get_weights()
    weights = np.array(weights)
    
    # put the mask into mask_array
    mask_array = []
    for mask_num in mask_nums:
        mask = masking(mask_num)  # masking method of bit_wise_And.py
        mask_array.append(mask)
    
    # get the model name and the name of each layer  * use to save weights file 
    model_name, layer_name = get_layer_name(model)
    
    # make directory  * use to save weights file    
    if not os.path.exists('./'+ model_name):
        os.mkdir('./'+ model_name)
    dir_origin = './{}/original/'.format(model_name)
    if not os.path.exists(dir_origin): 
        os.mkdir(dir_origin)

        # save the original weights
        layer_index = 0
        for weight in weights:
            weight = weight.flatten()
            filepath = dir_origin + layer_name[layer_index]
            np.save(filepath, weight)
            layer_index += 1
    
    # quantify start. for loop runs all mask
    for i in range(len(mask_array)):
        layer = []

        # make directories for modified weights
        dir_modified = './{}/bitwiseAnd_{}/'.format(model_name, mask_nums[i])
        if not os.path.exists(dir_modified): 
            os.mkdir(dir_modified)
        else:
            continue
        
        layer_index = 0
        # 'weight' variable is weights of each layer
        for weight in weights:
            # flatten method changes the multi dimension array to a one dimension
            weight_flatten = weight.flatten()
            # for loop runs all number 
            for num in range(len(weight_flatten)):
                # bit_wise_And method
                weight_flatten[num] = bit_wise_And(weight_flatten[num], mask_array[i])
            
            # save the modified weights
            filepath = dir_modified + layer_name[layer_index]
            np.save(filepath, weight_flatten)
            print("save complete : {}".format(filepath))
            layer_index += 1
            
            #reshape for saving model
            weight_flatten = weight_flatten.reshape(weight.shape)
            layer.append(weight_flatten)

        # set modified weighs 
        layer = np.array(layer)
        model.set_weights(layer)

        # serialize model to JSON
        model_json = model.to_json()
        with open(dir_modified+"model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights(dir_modified+"model.h5")
        print("Saved model to disk")