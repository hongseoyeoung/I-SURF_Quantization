import numpy as np
import struct
import binascii
import os
from model_info import get_layer_name

def convert_fl32_to_fl16(model):
    weights = model.get_weights()

    model_name, layer_name = get_layer_name(model)

    if not os.path.exists('./{}'.format(model_name)):
        os.mkdir('./{}'.format(model_name))

    if not os.path.exists('./{}/fl16'.format(model_name)):
        os.mkdir('./{}/fl16'.format(model_name))

    for i in range(len(weights)):
        weights[i] = np.array(weights[i], np.float16)
        np.save('./{}/fl16/{}'.format(model_name ,layer_name[i]), weights[i])

def convert_fl32_to_fix8(model):
    weights = model.get_weights()

    model_name, layer_name = get_layer_name(model)

    if not os.path.exists('./{}'.format(model_name)):
        os.mkdir('./{}'.format(model_name))

    if not os.path.exists('./{}/fix8'.format(model_name)):
        os.mkdir('./{}/fix8'.format(model_name))

    k = 0
    for j in range(len(weights)):
        weight_f = weights[j].flatten()
        for i in range(len(weight_f)):
            weight_f[i] = float32_to_fixed8(weight_f[i])
        weight_f = np.array(weight_f, np.int8)
        np.save('./{}/fix8/{}'.format(model_name ,layer_name[k]), weight_f)
        k += 1
        weights[j] = weight_f.reshape(weights[j].shape)

def convert_fix8_to_fl32(weights):
    tmp = np.zeros(weights.shape)
    for i in range(len(weights)):
        tmp[i] = fix8_to_float32(weights[i])
    return tmp

def float32_to_fixed8(x):
    a = struct.pack('>f', x)
    b = binascii.hexlify(a)
    float32 = int(b, 16)

    sign = (float32 >> 24) & 0x80
    exponent = ((float32 >> 23) & 0xff) - 127
    mentissa = (float32 & 0x007fffff) | 0x00800000

    if exponent > 0:
        origin = mentissa << exponent
    else:
        exponent *= -1
        origin = mentissa >> exponent
    
    if (origin >> 23) >= 2:
        fraction = 0x07f
    else:
        fraction = (origin >> (23-6)) & 0x7f
        
    return (sign | fraction)

def fix8_to_float32(x):
    sign = int((x >> 7) & 0x00000001)
    exponent = 0
    fraction = int(x & 0x0000007f)

    mentissa = fraction << (23-6)
    if fraction != 0:
        while not(mentissa & 0x00800000):
            mentissa = mentissa << 1
            exponent -= 1
        mentissa &= 0x007fffff
    else:
        exponent = -127

    exponent += 127
    tmp = struct.pack('I', (sign << 31) | (exponent << 23) | mentissa)
    f = struct.unpack('f', tmp)
    return f[0]