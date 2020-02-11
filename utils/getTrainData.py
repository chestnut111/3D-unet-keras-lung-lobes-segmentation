from keras.utils import to_categorical
import numpy as np 
from .utils import crop_brain_region, get_roi


def image_gen3d(ls,  batch_size=(48,224,224)):
     
    out_img = []
    out_mask = []

    while True:
        np.random.shuffle(ls)
        for data in ls:       
            img = data
            mask = data[:-10] + 'mask.nii.gz'
           
            img, mask, weight, original_shape, bbox = crop_brain_region([img], mask)
            img, mask = get_roi(img, mask)
                               
            out_img += [img]
            out_mask += [mask]                         
            if len(out_img) >= batch_size:
                yield np.stack(out_img, 0), to_categorical(np.stack(out_mask, 0))
                out_img, out_mask=[], [] 
                