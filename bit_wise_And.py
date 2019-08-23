import struct

# return mask_num. ex) if mask_num is 16, it returns '11111111111111110000000000000000'
def masking(mask_num):
    return ('0' * mask_num).rjust(32,'1')

# bit-wise And
def bit_wise_And(num, mask):
    result = []
    # struct.pack changes float number to 32-bit floating point number.
    # 'value' variable is divided into 4 parts
    value = struct.pack('!f', num)
    for i in range(4):
        # bit-wise And for each part
        result.append(int(mask[i*8:(i+1)*8], 2) & value[i])
    
    # return modified number
    return struct.unpack('!f', bytes(result))[0]


    