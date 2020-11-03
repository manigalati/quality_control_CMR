import os
import numpy as np
from PIL import Image
import nibabel as nib
from medpy.metric import binary

def resize_image_by_padding(image, new_shape, pad_value=None):
    shape = tuple(list(image.shape))
    new_shape = tuple(np.max(np.concatenate((shape, new_shape)).reshape((2, len(shape))), axis=0))
    if pad_value is None:
        if len(shape) == 2:
            pad_value = image[0, 0]
        elif len(shape) == 3:
            pad_value = image[0, 0, 0]
        else:
            raise ValueError("Image must be either 2 or 3 dimensional")
    res = np.ones(list(new_shape), dtype=image.dtype) * pad_value
    start = np.array(new_shape) / 2. - np.array(shape) / 2.
    if len(shape) == 2:
        res[int(start[0]):int(start[0]) + int(shape[0]), int(start[1]):int(start[1]) + int(shape[1])] = image
    elif len(shape) == 3:
        res[int(start[0]):int(start[0]) + int(shape[0]), int(start[1]):int(start[1]) + int(shape[1]),
        int(start[2]):int(start[2]) + int(shape[2])] = image
    return res

def center_crop_2D_image(img, crop_size):
    center = np.array(img.shape) / 2.
    if type(crop_size) not in (tuple, list):
        center_crop = [int(crop_size)] * len(img.shape)
    else:
        center_crop = crop_size
        assert len(center_crop) == len(
            img.shape), "If you provide a list/tuple as center crop make sure it has the same len as your data has dims (2d)"
    return img[int(center[0] - center_crop[0] / 2.):int(center[0] + center_crop[0] / 2.),
           int(center[1] - center_crop[1] / 2.):int(center[1] + center_crop[1] / 2.)]

def get_gt(src):
  gt={"ED":[],"ES":[]}
  for patient in os.listdir(src):
    if(os.path.isdir(src+patient)):
      frames=[]
      for f in os.listdir(src+patient):
          if("frame" in f and "_gt" not in f):
            frames.append(int(f.split("frame")[1].split(".nii.gz")[0]))
      scan_ED=nib.load(src+patient+"/"+patient+"_frame{:02d}_gt.nii.gz".format(min(frames)))
      scan_ES=nib.load(src+patient+"/"+patient+"_frame{:02d}_gt.nii.gz".format(max(frames)))
      scan_ED=scan_ED.get_fdata().transpose(2,0,1)
      scan_ES=scan_ES.get_fdata().transpose(2,0,1)
      for img in scan_ED:
        img=img.astype(int)
        img=resize_image_by_padding(img,(256,256))
        if(any(np.array(img.shape)>256)):
          img=center_crop_2D_image(img,(256,256))
        gt["ED"].append((patient,img))
      for img in scan_ES:
        img=img.astype(int)
        img=resize_image_by_padding(img,(256,256))
        if(any(np.array(img.shape)>256)):
          img=center_crop_2D_image(img,(256,256))
        gt["ES"].append((patient,img))
  return gt
  
def get_test(src):
  test={"ED":[],"ES":[]}
  for scan in os.listdir(src):
    if(scan.endswith("_ED.nii.gz")):
      patient=scan.split("_ED.nii.gz")[0]
      scan=nib.load(src+scan)
      scan=scan.get_fdata().transpose(2,0,1)
      for img in scan:
        img=img.astype(int)
        img=resize_image_by_padding(img,(256,256))
        if(any(np.array(img.shape)>256)):
          img=center_crop_2D_image(img,(256,256))
        test["ED"].append((patient,img))
    elif(scan.endswith("_ES.nii.gz")):
      patient=scan.split("_ES.nii.gz")[0]
      scan=nib.load(src+scan)
      scan=scan.get_fdata().transpose(2,0,1)
      for img in scan:
        img=img.astype(int)
        img=resize_image_by_padding(img,(256,256))
        if(any(np.array(img.shape)>256)):
          img=center_crop_2D_image(img,(256,256))
        test["ES"].append((patient,img))
  return test
  
def get_results(prediction,reference):
  results={}
  for c,key in enumerate(["","RV_","MYO_","LV_"]):
    ref=np.copy(reference)
    pred=np.copy(prediction)

    ref=ref if c==0 else np.where(ref!=c,0,ref)
    pred=pred if c==0 else np.where(np.rint(pred)!=c,0,pred)
    
    results[key+"SED"]=np.sum((ref-pred)**2)
    results[key+"SED_rint"]=np.sum((ref-np.rint(pred))**2)
    results[key+"maxSED"]=np.sum(((ref-pred)**2).reshape(ref.shape[0],-1),axis=1).max()
    results[key+"maxSED_rint"]=np.sum(((ref-np.rint(pred))**2).reshape(ref.shape[0],-1),axis=1).max()
    results[key+"Dice"]=2*np.sum(pred*np.where(ref!=0,1,0))/np.sum(pred+np.where(ref!=0,1,0)) if np.sum(pred+np.where(ref!=0,1,0))!=0 else 0
    results[key+"Dice_rint"]=binary.dc(np.where(ref!=0,1,0),np.where(np.rint(pred)!=0,1,0))
    try:
      results[key+"HD_rint"]=binary.hd(np.where(ref!=0,1,0),np.where(np.rint(pred)!=0,1,0))
    except Exception:
      results[key+"HD_rint"]=np.nan
  return results
  
def display_image(img):
  img=np.rint(img)
  img=np.rint(img/3*255)
  display(Image.fromarray(img.astype(np.uint8)))

def display_difference(prediction,reference):
  difference=np.zeros([256,256,3])
  difference=np.zeros([256,256,3])
  difference[np.rint(prediction)!=np.rint(reference)]=[240,52,52]
  difference[np.rint(prediction)==np.rint(reference)]=(np.abs(prediction-reference)[np.rint(prediction)==np.rint(reference)]*np.array([[240,52,52]]).T).T
  display(Image.fromarray(difference.astype(np.uint8)))