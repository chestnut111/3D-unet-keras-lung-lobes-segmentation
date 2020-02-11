import numpy as np
import random
import os
import SimpleITK as sitk
import pickle
from scipy import ndimage
import configTrain
import copy


def crop_brain_region(imgs, mask, with_gt=True):
    
    WINDOW_LEVEL = (1000,-500)
    volume_list = []
    for idx, mod in enumerate(imgs):    
        volume = sitk.ReadImage(mod)
        img_window = sitk.Cast(sitk.IntensityWindowing(volume,
                                          windowMinimum=WINDOW_LEVEL[1] - WINDOW_LEVEL[0] / 2.0,
                                          windowMaximum=WINDOW_LEVEL[1] + WINDOW_LEVEL[0] / 2.0),
                                          sitk.sitkUInt8)   
        img_window_arr =  sitk.GetArrayFromImage(img_window)
        if idx == 0: 
            original_shape = img_window_arr.shape #return 
            margin = 1 
            bbmin, bbmax = get_none_zero_region(img_window_arr, margin)
            weight = np.asarray(img_window_arr > 0, np.float32) #return 
        img_window_arr_crop = crop_ND_volume_with_bounding_box(img_window_arr, bbmin, bbmax)
        img_window_arr_crop_norm = itensity_normalize_one_volume(img_window_arr_crop)
        volume_list.append(img_window_arr_crop_norm)
    if with_gt:
        label = load_nifty_volume_as_array(mask)

        label[label == 1] = 0
        label[label == 4] = 1
        label[label == 5] = 2
        label[label == 6] = 3
        label[label == 7] = 4
        label[label == 8] = 5
        label[label > 10] = 0
    
        label = crop_ND_volume_with_bounding_box(label, bbmin, bbmax)
        return volume_list, label, weight, original_shape, [bbmin, bbmax]
    else:
        return volume_list, None, weight, original_shape, [bbmin, bbmax]
    
def get_roi(img, mask):
    volume_shape = img[0].shape
    center_point = get_random_roi_sampling_center(volume_shape, configTrain.PATCH_SIZE, configTrain.center)    
    sub_label = extract_roi_from_volume(mask,
                                        center_point,
                                        configTrain.PATCH_SIZE,
                                        fill = 'zero')
    
    sub_img = extract_roi_from_volume(img[0],
                                        center_point,
                                        configTrain.PATCH_SIZE,
                                        fill = 'zero') 
    
    axis = [1,2,3,0]
    sub_data  = np.transpose([sub_img], axis)
    sub_label  = np.transpose(sub_label[np.newaxis, ...], axis)    
    return  sub_data, sub_label    

        
def get_random_roi_sampling_center(input_shape, output_shape, sample_mode='full'):
    """
    get a random coordinate representing the center of a roi for sampling
    inputs:
        input_shape: the shape of sampled volume
        output_shape: the desired roi shape
        sample_mode: 'valid': the entire roi should be inside the input volume
                     'full': only the roi centre should be inside the input volume
        bounding_box: the bounding box which the roi center should be limited to
    outputs:
        center: the output center coordinate of a roi
    """
    center = []
    for i in range(len(input_shape)):
        if sample_mode == 'full':
            x0 = 0
            x1 = input_shape[i]
        elif sample_mode == 'valid':
            x0 = int(output_shape[i]/2)   
            x1 = input_shape[i] - x0
            
        if x1 <= x0:  #??
            centeri = int((x0 + x1)/2)
        else:
            centeri = random.randint(x0, x1)
        center.append(centeri)
    return center

def crop_ND_volume_with_bounding_box(volume, min_idx, max_idx):
    """
    crop/extract a subregion form an nd image.
    """
    dim = len(volume.shape)
    assert(dim >= 2 and dim <= 5)
    if(dim == 2):
        output = volume[np.ix_(range(min_idx[0], max_idx[0] + 1),
                               range(min_idx[1], max_idx[1] + 1))]
    elif(dim == 3):
        output = volume[np.ix_(range(min_idx[0], max_idx[0] + 1),
                               range(min_idx[1], max_idx[1] + 1),
                               range(min_idx[2], max_idx[2] + 1))]
    elif(dim == 4):
        output = volume[np.ix_(range(min_idx[0], max_idx[0] + 1),
                               range(min_idx[1], max_idx[1] + 1),
                               range(min_idx[2], max_idx[2] + 1),
                               range(min_idx[3], max_idx[3] + 1))]
    elif(dim == 5):
        output = volume[np.ix_(range(min_idx[0], max_idx[0] + 1),
                               range(min_idx[1], max_idx[1] + 1),
                               range(min_idx[2], max_idx[2] + 1),
                               range(min_idx[3], max_idx[3] + 1),
                               range(min_idx[4], max_idx[4] + 1))]
    else:
        raise ValueError("the dimension number shoud be 2 to 5")
    return output

def extract_roi_from_volume(volume, in_center, output_shape, fill = 'random'):
    """
    extract a roi from a 3d volume
    inputs:
        volume: the input 3D volume
        in_center: the center of the roi
        output_shape: the size of the roi
        fill: 'random' or 'zero', the mode to fill roi region where is outside of the input volume
    outputs:
        output: the roi volume
    """
    input_shape = volume.shape   
    if(fill == 'random'):
        output = np.random.normal(0, 1, size = output_shape)
    else:
        output = np.zeros(output_shape)
    r0max = [int(x/2) for x in output_shape]
    r1max = [output_shape[i] - r0max[i] for i in range(len(r0max))]
    r0 = [min(r0max[i], in_center[i]) for i in range(len(r0max))]
    r1 = [min(r1max[i], input_shape[i] - in_center[i]) for i in range(len(r0max))]
    out_center = r0max

    output[np.ix_(range(out_center[0] - r0[0], out_center[0] + r1[0]),
                  range(out_center[1] - r0[1], out_center[1] + r1[1]),
                  range(out_center[2] - r0[2], out_center[2] + r1[2]))] = \
        volume[np.ix_(range(in_center[0] - r0[0], in_center[0] + r1[0]),
                      range(in_center[1] - r0[1], in_center[1] + r1[1]),
                      range(in_center[2] - r0[2], in_center[2] + r1[2]))]
    return output

def load_nifty_volume_as_array(filename):
    """
    load nifty image into numpy array, and transpose it based on the [z,y,x] axis order
    The output array shape is like [Depth, Height, Width]
    inputs:
        filename: the input file name, should be *.nii or *.nii.gz
    outputs:
        data: a numpy data array
    """
    img = sitk.ReadImage(filename)
    img_arr = sitk.GetArrayFromImage(img)
    return img_arr

def itensity_normalize_one_volume(volume):
    """
    normalize the itensity of an nd volume based on the mean and std of nonzeor region
    inputs:
        volume: the input nd volume
    outputs:
        out: the normalized nd volume
    """
    pixels = volume[volume > 0]
    mean = pixels.mean()
    std  = pixels.std()
    out = (volume - mean)/std
    out_random = np.zeros(volume.shape)
    out[volume == 0] = out_random[volume == 0]
    return out         
                
def get_ND_bounding_box(label, margin):
    """
    get the bounding box of the non-zero region of an ND volume
    """
    input_shape = label.shape
    if(type(margin) is int ):
        margin = [margin]*len(input_shape)
    assert(len(input_shape) == len(margin))
    indxes = np.nonzero(label)
    idx_min = []
    idx_max = []
    for i in range(len(input_shape)):
        idx_min.append(indxes[i].min())
        idx_max.append(indxes[i].max())

    for i in range(len(input_shape)):
        idx_min[i] = max(idx_min[i] - margin[i], 0)
        idx_max[i] = min(idx_max[i] + margin[i], input_shape[i] - 1)
    return idx_min, idx_max

def set_ND_volume_roi_with_bounding_box_range(volume, bb_min, bb_max, sub_volume):
    """
    set a subregion to an nd image.
    """
    dim = len(bb_min)
    out = volume
    if(dim == 2):
        out[np.ix_(range(bb_min[0], bb_max[0] + 1),
                   range(bb_min[1], bb_max[1] + 1))] = sub_volume
    elif(dim == 3):
        out[np.ix_(range(bb_min[0], bb_max[0] + 1),
                   range(bb_min[1], bb_max[1] + 1),
                   range(bb_min[2], bb_max[2] + 1))] = sub_volume
    elif(dim == 4):
        out[np.ix_(range(bb_min[0], bb_max[0] + 1),
                   range(bb_min[1], bb_max[1] + 1),
                   range(bb_min[2], bb_max[2] + 1),
                   range(bb_min[3], bb_max[3] + 1))] = sub_volume
    else:
        raise ValueError("array dimension should be 2, 3 or 4")
    return out

def set_roi_to_volume(volume, center, sub_volume):
    """
    set the content of an roi of a 3d/4d volume to a sub volume
    inputs:
        volume: the input 3D/4D volume
        center: the center of the roi
        sub_volume: the content of sub volume
    outputs:
        output_volume: the output 3D/4D volume
    """
    volume_shape = volume.shape   
    patch_shape = sub_volume.shape
    output_volume = volume
    for i in range(len(center)):
        if(center[i] >= volume_shape[i]):
            return output_volume
    r0max = [int(x/2) for x in patch_shape]
    r1max = [patch_shape[i] - r0max[i] for i in range(len(r0max))]
    r0 = [min(r0max[i], center[i]) for i in range(len(r0max))]
    r1 = [min(r1max[i], volume_shape[i] - center[i]) for i in range(len(r0max))]
    patch_center = r0max

    if(len(center) == 3):
        output_volume[np.ix_(range(center[0] - r0[0], center[0] + r1[0]),
                             range(center[1] - r0[1], center[1] + r1[1]),
                             range(center[2] - r0[2], center[2] + r1[2]))] = \
            sub_volume[np.ix_(range(patch_center[0] - r0[0], patch_center[0] + r1[0]),
                              range(patch_center[1] - r0[1], patch_center[1] + r1[1]),
                              range(patch_center[2] - r0[2], patch_center[2] + r1[2]))]
    elif(len(center) == 4):
        output_volume[np.ix_(range(center[0] - r0[0], center[0] + r1[0]),
                             range(center[1] - r0[1], center[1] + r1[1]),
                             range(center[2] - r0[2], center[2] + r1[2]),
                             range(center[3] - r0[3], center[3] + r1[3]))] = \
            sub_volume[np.ix_(range(patch_center[0] - r0[0], patch_center[0] + r1[0]),
                              range(patch_center[1] - r0[1], patch_center[1] + r1[1]),
                              range(patch_center[2] - r0[2], patch_center[2] + r1[2]),
                              range(patch_center[3] - r0[3], patch_center[3] + r1[3]))]
    else:
        raise ValueError("array dimension should be 3 or 4")        
    return output_volume  

def get_none_zero_region(im, margin):
    """
    get the bounding box of the non-zero region of an ND volume
    """
    input_shape = im.shape
    if(type(margin) is int ):
        margin = [margin]*len(input_shape)
    assert(len(input_shape) == len(margin))
    indxes = np.nonzero(im)
    idx_min = []
    idx_max = []
    for i in range(len(input_shape)):
        idx_min.append(indxes[i].min())
        idx_max.append(indxes[i].max())

    for i in range(len(input_shape)):
        idx_min[i] = max(idx_min[i] - margin[i], 0)
        idx_max[i] = min(idx_max[i] + margin[i], input_shape[i] - 1)
    return idx_min, idx_max


