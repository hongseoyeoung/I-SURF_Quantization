import numpy as np
from model_info import get_layer_name
from convert import convert_fix8_to_fl32

def swap_layer(model, layer_num, quanti_type):
    tmp_model = model
    # get the all weights
    weights = tmp_model.get_weights()
    weights = np.array(weights)

    # get the model name and the name of each layer
    model_name, layer_name = get_layer_name(model)
    
    # find the layer index
    kernel_num = layer_num * 2
    bias_num = layer_num * 2 + 1

    # find weights file
    kernel_path = './{}/{}/{}.npy'.format(model_name, quanti_type, layer_name[kernel_num])
    bias_path = './{}/{}/{}.npy'.format(model_name, quanti_type, layer_name[bias_num])


    # load layer' weights
    load_kernel = np.load(kernel_path)
    load_bias = np.load(bias_path)

    if quanti_type == 'fix8':
        load_kernel = convert_fix8_to_fl32(load_kernel)
        load_bias = convert_fix8_to_fl32(load_bias)

    # reshape for setting
    load_kernel = load_kernel.reshape(weights[kernel_num].shape)
    load_bias = load_bias.reshape(weights[bias_num].shape)

    # apply values
    weights[kernel_num] = load_kernel
    weights[bias_num] = load_bias
    
    # set model
    tmp_model.set_weights(weights)
    return tmp_model
