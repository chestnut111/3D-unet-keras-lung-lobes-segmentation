# 3DUnet-keras


3D Unet biomedical segmentation model powered by tensorpack with fast io speed.

Borrow a lot of codes from https://github.com/tkuanlun350/3DUnet-Tensorflow-Brats18. 
I streamlined the code and changed it to the keras version.

>I want to verify the effectiveness (consistent improvement despite of slight implementation differences and different deep-learning framework) of some architecture proposed these years. Such as
dice_loss, generalised dice_loss, residual connection, instance norm, deep supervision ...etc. Those design are popular and used in many papers in BRATS competition.  



## Dependencies
- Python 3; 
- TensorFlow 1.12.0;
- Keras 2.2.4;

+ (Optional) If you want to use [Bias Correction](https://ieeexplore.ieee.org/abstract/document/5445030/) 
you have to install nipype and ANTs (see preprocess.py)

```
DIR/
  training/
    HGG/
    LGG/
  val/
    BRATS*.nii.gz
```

## Data 
2019 training data
链接:https://pan.baidu.com/s/13va5fj-Yr2xcnjVyYhkJwQ  
密码:nfah

