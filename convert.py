import numpy as np
import struct
import binascii
import os
from model_info import get_layer_name

# convert float32 to float16
def convert_fl32_to_fl16(model):
    # get model's weights
    weights = model.get_weights()
    # get model's name
    model_name, layer_name = get_layer_name(model)

    # make the directory
    if not os.path.exists('./{}'.format(model_name)):
        os.mkdir('./{}'.format(model_name))

    if not os.path.exists('./{}/fl16'.format(model_name)):
        os.mkdir('./{}/fl16'.format(model_name))

    # change to float16 and save
    for i in range(len(weights)):
        weights[i] = np.array(weights[i], np.float16)
        np.save('./{}/fl16/{}'.format(model_name ,layer_name[i]), weights[i])

# convert float32 to fix8
def convert_fl32_to_fix8(model):
    # get the model's  weights
    weights = model.get_weights()
    # get the model's name
    model_name, layer_name = get_layer_name(model)
    # make the directory 
    if not os.path.exists('./{}'.format(model_name)):
        os.mkdir('./{}'.format(model_name))

    if not os.path.exists('./{}/fix8'.format(model_name)):
        os.mkdir('./{}/fix8'.format(model_name))

    # change to fixed 8bit
    k = 0
    for j in range(len(weights)):
        weight_f = weights[j].flatten()
        for i in range(len(weight_f)):
            # convert method
            weight_f[i] = float32_to_fixed8(weight_f[i])
        # saving to 8bit variable
        weight_f = np.array(weight_f, np.int8)
        # save weights
        np.save('./{}/fix8/{}'.format(model_name ,layer_name[k]), weight_f)
        k += 1

# conver to fixed 8 to float32
def convert_fix8_to_fl32(weights):
    # create temp variable 
    tmp = np.zeros(weights.shape)
    # change
    for i in range(len(weights)):
        tmp[i] = fix8_to_float32(weights[i])
    return tmp

# change float32 to fixed8
def float32_to_fixed8(x):
    # struct.pack changes float number(x) to 32-bit floating point number.
    a = struct.pack('>f', x)
    b = binascii.hexlify(a)
    float32 = int(b, 16)

    # extract the value of each part using a bit shift and bit-wise and
    sign = (float32 >> 24) & 0x80
    exponent = ((float32 >> 23) & 0xff) - 127
    mentissa = (float32 & 0x007fffff) | 0x00800000

    # change floating point to fixed point
    # This fixed point is 1 sign bit, 1 integer bit, 6 fraction bit
    if exponent > 0:
        origin = mentissa << exponent
    else:
        exponent *= -1
        origin = mentissa >> exponent
    
    # overflow processing
    # this fixed point' max is 1.984. so if integer part is larger than 2, fraction is 01111111 that is 1.984375
    if (origin >> 23) >= 2:
        fraction = 0x07f
    else:
        fraction = (origin >> (23-6)) & 0x7f
    # return bitwise or
    return (sign | fraction)

# change fixed 8 to float32
# this fixed point cannot be uesed as it is, so it needs to be replaced with a 32bit floating point.
def fix8_to_float32(x):
    # extract the value of each part using a bit shift and bit-wise and
    sign = int((x >> 7) & 0x00000001)
    exponent = 0
    fraction = int(x & 0x0000007f)

    # floating processing
    mentissa = fraction << (23-6)
    if fraction != 0:
        while not(mentissa & 0x00800000):
            mentissa = mentissa << 1
            exponent -= 1
        mentissa &= 0x007fffff
    else:
        exponent = -127

    # plus bias
    exponent += 127
    # bitwise or
    tmp = struct.pack('I', (sign << 31) | (exponent << 23) | mentissa)
    f = struct.unpack('f', tmp)
    return f[0]