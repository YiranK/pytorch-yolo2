#!/usr/bin/python
# encoding: utf-8
import random
import os
from PIL import Image
import numpy as np


def scale_image_channel(im, c, v):
    cs = list(im.split())
    cs[c] = cs[c].point(lambda i: i * v)
    out = Image.merge(im.mode, tuple(cs))
    return out

def distort_image(im, hue, sat, val):
    im = im.convert('HSV')
    cs = list(im.split())
    cs[1] = cs[1].point(lambda i: i * sat)
    cs[2] = cs[2].point(lambda i: i * val)
    
    def change_hue(x):
        x += hue*255
        if x > 255:
            x -= 255
        if x < 0:
            x += 255
        return x
    cs[0] = cs[0].point(change_hue)
    im = Image.merge(im.mode, tuple(cs))

    im = im.convert('RGB')
    #constrain_image(im)
    return im

def rand_scale(s):
    scale = random.uniform(1, s)
    if(random.randint(1,10000)%2): 
        return scale
    return 1./scale

def random_distort_image(im, hue, saturation, exposure):
    dhue = random.uniform(-hue, hue)
    dsat = rand_scale(saturation)
    dexp = rand_scale(exposure)
    res = distort_image(im, dhue, dsat, dexp)
    return res

def data_augmentation(img, shape, jitter, hue, saturation, exposure):
    oh = img.height  
    ow = img.width
    
    dw =int(ow*jitter)
    dh =int(oh*jitter)

    pleft  = random.randint(-dw, dw)
    pright = random.randint(-dw, dw)
    ptop   = random.randint(-dh, dh)
    pbot   = random.randint(-dh, dh)

    swidth =  ow - pleft - pright
    sheight = oh - ptop - pbot

    sx = float(swidth)  / ow
    sy = float(sheight) / oh
    
    flip = random.randint(1,10000)%2
    cropped = img.crop( (pleft, ptop, pleft + swidth - 1, ptop + sheight - 1))

    dx = (float(pleft)/ow)/sx
    dy = (float(ptop) /oh)/sy

    sized = cropped.resize(shape)

    if flip: 
        sized = sized.transpose(Image.FLIP_LEFT_RIGHT)
    img = random_distort_image(sized, hue, saturation, exposure)
    
    return img, flip, dx,dy,sx,sy 

def fill_truth_detection(labpath, w, h, flip, dx, dy, sx, sy):
    max_boxes = 50
    label = np.zeros((max_boxes,1+4+723)) # one value indicating 609 class, 4 box, 723 attribute, 20180509
    if os.path.getsize(labpath):
        bs = np.load(labpath) # for '.npy', 20180509
        # print labpath
        if bs is None:
            return label
        #bs = np.reshape(bs, (-1, 6)) # no use, 20180509

        cc = 0
        for i in range(50):
            if bs[i][0] != 1:
                break
            # get only one class index of bs[i], 20180510
            cls_idx = bs[i][2].nonzero()[0][0]

            # if i < 3:
            #     print (i,' origin bbox:', bs[i][1])

            x_center_float = bs[i][1][0]*(1./w) #x
            y_center_float = bs[i][1][1]*(1./h) #y
            w_float = bs[i][1][2]*(1./w) #w
            h_float = bs[i][1][3]*(1./h) #h
            bbox = np.array([cls_idx,x_center_float,y_center_float,w_float,h_float])

            # if i < 3:
            #     print (i,' class:', bs[i][2].nonzero())
            #     print (i,' /wh bbox:', bs[i][1])
            #     print (i,' cls concat /wh box:', bbox)

            # origin YOLOv2 handle PASCAL VOC label file
            # x1 = bbox[1] - bbox[3]/2
            # y1 = bbox[2] - bbox[4]/2
            # x2 = bbox[1] + bbox[3]/2
            # y2 = bbox[2] + bbox[4]/2

            # bbox:[x1,y1,w,h] -> [x1,y1,x2,y2], 20180511
            x1 = bbox[1]
            y1 = bbox[2]
            x2 = bbox[1] + bbox[3] # x1+w
            y2 = bbox[2] + bbox[4] # y1+h
            
            x1 = min(0.999, max(0, x1 * sx - dx)) 
            y1 = min(0.999, max(0, y1 * sy - dy)) 
            x2 = min(0.999, max(0, x2 * sx - dx))
            y2 = min(0.999, max(0, y2 * sy - dy))

            # if i < 3:
            #     print(i,' (x1,y1,x2,y2):',x1,y1,x2,y2)
            
            bbox[1] = (x1 + x2)/2
            bbox[2] = (y1 + y2)/2
            bbox[3] = (x2 - x1)
            bbox[4] = (y2 - y1)

            # if i<3:
            #     print (i,' after cropped (x,y,w,h):', bbox)

            if flip:
                bbox[1] =  0.999 - bbox[1]
            
            if bbox[3] < 0.001 or bbox[4] < 0.001:
                continue

            label[cc] = np.concatenate((bbox,bs[i][3]))

            # if i<3:
            #     print (i,' label:',label[cc])

            cc += 1
            if cc >= 50:
                break
    label = np.reshape(label, (-1))
    return label

def load_data_detection(imgpath, shape, jitter, hue, saturation, exposure):
    # for vg_train.txt has the img id but not img path, modify the labpath and imgpath, 20180509
    labpath = os.path.join('/mnt/lustre/kangyiran2/zero-shot-detection/dataset/zsd_anno', imgpath + '.npy')
    imgpath = os.path.join('/mnt/lustre/kangyiran2/zero-shot-detection/dataset/imgs/VG_100K', imgpath + '.jpg')
    #labpath = imgpath.replace('images', 'labels').replace('JPEGImages', 'labels').replace('.jpg', '.npy').replace('.png','.npy')

    ## data augmentation
    img = Image.open(imgpath).convert('RGB')
    img,flip,dx,dy,sx,sy = data_augmentation(img, shape, jitter, hue, saturation, exposure)
    label = fill_truth_detection(labpath, img.width, img.height, flip, dx, dy, 1./sx, 1./sy)
    return img,label
