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

## create models
### create bitwise And model
```
python create_bitwise_model.py
```

### create float16 model
```
python create_float16_model.py
```

### create fix8 model
```
python create_fix8_model.py
```


 
