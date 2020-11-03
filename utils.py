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