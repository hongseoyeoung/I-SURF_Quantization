def get_layer_name(model):
    # get the model name and the name of each layer
    model_name = model.get_config()['name']
    layer_name = []
    layer_config = model.get_config()['layers']
    for layer in layer_config:
        if layer['class_name'] == 'Conv2D' or layer['class_name'] == 'Dense':
            layer_name.append(layer['name']+'_kernel')
            layer_name.append(layer['name']+'_bias')

    return model_name, layer_name