# I-SURF_Quantization
### Purpose
Reduce storage and memory and improve computational speed by reducing or changing to fixed point the bits of weights in CNN models 

### Prerequisites                               
* keras 2.2.4                    
* matplotlib  3.1.1                                  
* pillow 6.1.0                                   
* python 3.6.8               
* tensorflow-gpu 1.12.0   

### Usage

#### 1. Download the repository

```
git clone "https://github.com/hongseoyeoung/I-SURF_Quantization"
```

#### 2. Set condition
1. install tensorflow
```
pip install tensorflow
```
if you want to install tensorflow-gpu
```
pip install tensorflow-gpu
```
2. install keras
```
pip install keras
```
3. install pillow
```
pip install pillow
```
if you use an anaconda virtual environment
```
cuda install pillow
```

### 3. Run

### prediction check of vgg19
if you want to change CNN model, you can. but only keras model

```
python prediction_check.py
```
coffee_mug (89.09%) cup (5.26%) coffeepot (2.26%)

## check models

### create bitwise And model
```
python create_bitwise_model.py
```
it makes model.json, model.h5 and weights file(.npy) of each layer and each bit-wise And

### create float16 model
```
python create_float16_model.py
```
it makes float 16bit weights file(.npy)

### create fix8 model
```
python create_fix8_model.py
```
it makes fixed 8 bit weights file(.npy)

## Check accuracy models

### check bit-wise And
```
python check bitwise_And.py
```
it makes test.txt
it checks the model's accuracy whose all layers are applied the same bit-wise And. 
To run this file, you must first run create_bitwise_model.py

### check bit-wise And each_layer
```
python check_bitwise_each_layer.py
```
it makes check_each_layer.txt
it checks model's accuracy that modified only one layer by each bit-wise And mask. 
To run this file, you must first run create_bitwise_model.py

check_each_layer.txt 
```
13 layer, 1 bit : coffee_mug (89.09%) cup (5.26%) coffeepot (2.26%)
13 layer, 2 bit : coffee_mug (89.09%) cup (5.26%) coffeepot (2.26%)
13 layer, 4 bit : coffee_mug (89.09%) cup (5.26%) coffeepot (2.26%)
13 layer, 8 bit : coffee_mug (89.09%) cup (5.26%) coffeepot (2.26%)
13 layer, 16 bit : coffee_mug (89.04%) cup (5.28%) coffeepot (2.28%)
13 layer, 17 bit : coffee_mug (88.98%) cup (5.29%) coffeepot (2.29%)
13 layer, 18 bit : coffee_mug (88.86%) cup (5.31%) coffeepot (2.32%)
13 layer, 19 bit : coffee_mug (88.72%) cup (5.33%) coffeepot (2.33%)
13 layer, 20 bit : coffee_mug (88.10%) cup (5.51%) coffeepot (2.47%)
13 layer, 21 bit : coffee_mug (87.36%) cup (5.75%) coffeepot (2.62%)
13 layer, 22 bit : coffee_mug (85.38%) cup (6.31%) coffeepot (2.94%)

14 layer, 1 bit : coffee_mug (89.09%) cup (5.26%) coffeepot (2.26%)
14 layer, 2 bit : coffee_mug (89.09%) cup (5.26%) coffeepot (2.26%)
14 layer, 4 bit : coffee_mug (89.09%) cup (5.26%) coffeepot (2.26%)
14 layer, 8 bit : coffee_mug (89.09%) cup (5.26%) coffeepot (2.26%)
14 layer, 16 bit : coffee_mug (89.06%) cup (5.27%) coffeepot (2.27%)
14 layer, 17 bit : coffee_mug (89.00%) cup (5.29%) coffeepot (2.28%)
14 layer, 18 bit : coffee_mug (88.94%) cup (5.30%) coffeepot (2.29%)
14 layer, 19 bit : coffee_mug (88.68%) cup (5.38%) coffeepot (2.35%)
14 layer, 20 bit : coffee_mug (88.34%) cup (5.50%) coffeepot (2.41%)
14 layer, 21 bit : coffee_mug (87.52%) cup (5.70%) coffeepot (2.54%)
14 layer, 22 bit : coffee_mug (85.46%) cup (6.09%) coffeepot (2.88%)

15 layer, 1 bit : coffee_mug (89.09%) cup (5.26%) coffeepot (2.26%)
15 layer, 2 bit : coffee_mug (89.09%) cup (5.26%) coffeepot (2.26%)
15 layer, 4 bit : coffee_mug (89.09%) cup (5.26%) coffeepot (2.26%)
15 layer, 8 bit : coffee_mug (89.09%) cup (5.26%) coffeepot (2.26%)
15 layer, 16 bit : coffee_mug (89.01%) cup (5.30%) coffeepot (2.28%)
15 layer, 17 bit : coffee_mug (88.90%) cup (5.32%) coffeepot (2.30%)
15 layer, 18 bit : coffee_mug (88.83%) cup (5.34%) coffeepot (2.32%)
15 layer, 19 bit : coffee_mug (88.65%) cup (5.40%) coffeepot (2.34%)
15 layer, 20 bit : coffee_mug (88.01%) cup (5.50%) coffeepot (2.54%)
15 layer, 21 bit : coffee_mug (86.30%) cup (5.88%) coffeepot (2.96%)
15 layer, 22 bit : coffee_mug (83.59%) cup (6.44%) coffeepot (3.50%)

```

### check specific
```
python chekc_specifi.py
```
it makes check_specific.txt
it can change each layer to the weight you want.
like this:
```
quanti_array = ['fix8','fix8','fix8','fix8','fl16','fix8'
                ,'fl16','fl16','bitwiseAnd_16','fl16','fl16','fl16'
                  ,'fl16','fl16','fl16','bitwiseAnd_22','fl16','fl16','fl16']
```
you can use fix8, fl16, bitwiseAnd_1,2,4,8,16,17,18,19,20,21,22

check_specific.txt:
```
0~0 layer, fix8 bit : coffee_mug (88.58%) cup (5.47%) coffeepot (2.23%)
0~1 layer, fix8 bit : coffee_mug (86.24%) cup (6.31%) coffeepot (2.34%)
0~2 layer, fix8 bit : coffee_mug (77.02%) cup (7.64%) coffeepot (4.11%)
0~3 layer, fix8 bit : coffee_mug (76.32%) cup (8.68%) coffeepot (3.81%)
0~4 layer, fl16 bit : coffee_mug (76.32%) cup (8.68%) coffeepot (3.82%)
0~5 layer, fix8 bit : coffee_mug (75.62%) cup (8.22%) coffeepot (2.34%)
0~6 layer, fl16 bit : coffee_mug (75.62%) cup (8.22%) coffeepot (2.34%)
0~7 layer, fl16 bit : coffee_mug (75.62%) cup (8.22%) coffeepot (2.34%)
0~8 layer, fl16 bit : coffee_mug (75.62%) cup (8.22%) coffeepot (2.34%)
0~9 layer, fl16 bit : coffee_mug (75.63%) cup (8.22%) coffeepot (2.34%)
0~10 layer, bitwiseAnd_21 bit : coffee_mug (73.99%) cup (8.07%) teapot (2.38%)
0~11 layer, fl16 bit : coffee_mug (73.99%) cup (8.07%) teapot (2.38%)
0~12 layer, bitwiseAnd_17 bit : coffee_mug (73.88%) cup (8.05%) teapot (2.38%)
0~13 layer, bitwiseAnd_17 bit : coffee_mug (73.77%) cup (8.04%) teapot (2.39%)
0~14 layer, bitwiseAnd_17 bit : coffee_mug (73.68%) cup (8.04%) teapot (2.40%)
0~15 layer, bitwiseAnd_17 bit : coffee_mug (73.24%) cup (8.08%) teapot (2.43%)
0~16 layer, bitwiseAnd_17 bit : coffee_mug (72.96%) cup (8.11%) teapot (2.45%)
0~17 layer, bitwiseAnd_17 bit : coffee_mug (72.54%) cup (8.14%) teapot (2.48%)
0~18 layer, bitwiseAnd_18 bit : coffee_mug (71.65%) cup (8.27%) teapot (2.51%)
```
That first line means the accuracy of applying only fix8 to the first layer.
That second line means the accuaracy of applying fix8 to the first layer and fix8d to the second layer.

It is applied one by one to the next layer. To check the sensitivity of each layer.
