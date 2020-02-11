# -*- coding: utf-8 -*-
# File: eval.py
import tqdm
import os
import numpy as np
from tensorpack.utils.utils import get_tqdm_kwargs
from .utils import *

def segment_one_image(data, model_func):
    """
    perform inference and unpad the volume to original shape
    """
    img, _, weight, original_shape, bbox = crop_brain_region([data], None, False)     
    temp_size = original_shape
    temp_bbox = bbox
    prob = batch_segmentation(img, model_func, data_shape = configTrain.PATCH_SIZE)
    pred = np.argmax(prob, axis=-1)
    out_label = np.asarray(pred, np.int16)
    final_label = np.zeros(temp_size, np.int16)
    final_label = set_ND_volume_roi_with_bounding_box_range(final_label, temp_bbox[0], temp_bbox[1], out_label)
    return final_label

def batch_segmentation(temp_imgs, model, data_shape=[19, 180, 160]):
    batch_size = configTrain.BATCH_SIZE
    data_channel = 1
    class_num = configTrain.NUM_CLASS
    image_shape = temp_imgs[0].shape
    label_shape = [data_shape[0], data_shape[1], data_shape[2]]
    D, H, W = image_shape
    input_center = [int(D/2), int(H/2), int(W/2)]
    temp_prob1 = np.zeros([D, H, W, class_num])

    sub_image_batches = []
    for center_slice in range(int(label_shape[0]/2), D + int(label_shape[0]/2), label_shape[0]):
        center_slice = min(center_slice, D - int(label_shape[0]/2))
        sub_image_batch = []
        for chn in range(data_channel):
            temp_input_center = [center_slice, input_center[1], input_center[2]]
            sub_image = extract_roi_from_volume(
                            temp_imgs[chn], temp_input_center, data_shape, fill="zero")
            sub_image_batch.append(sub_image)
        sub_image_batch = np.asanyarray(sub_image_batch, np.float32) 
        sub_image_batches.append(sub_image_batch) 
    
    total_batch = len(sub_image_batches)
    max_mini_batch = int((total_batch+batch_size-1)/batch_size)
    sub_label_idx1 = 0
    for mini_batch_idx in range(max_mini_batch):
        data_mini_batch = sub_image_batches[mini_batch_idx*batch_size:
                                      min((mini_batch_idx+1)*batch_size, total_batch)]
        if(mini_batch_idx == max_mini_batch - 1):
            for idx in range(batch_size - (total_batch - mini_batch_idx*batch_size)):
                data_mini_batch.append(np.zeros([data_channel] + list(data_shape)))
        data_mini_batch = np.asarray(data_mini_batch, np.float32)
        data_mini_batch = np.transpose(data_mini_batch, [0, 2, 3, 4, 1])
        prob_mini_batch1 = model.predict_on_batch(data_mini_batch)
        
        for batch_idx in range(prob_mini_batch1.shape[0]):
            center_slice = sub_label_idx1*label_shape[0] + int(label_shape[0]/2)
            center_slice = min(center_slice, D - int(label_shape[0]/2))
            temp_input_center = [center_slice, input_center[1], input_center[2], int(class_num/2)]
            sub_prob = np.reshape(prob_mini_batch1[batch_idx], label_shape + [class_num])
            temp_prob1 = set_roi_to_volume(temp_prob1, temp_input_center, sub_prob)
            sub_label_idx1 = sub_label_idx1 + 1
    
    return temp_prob1

