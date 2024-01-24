import os
import cv2
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from transformers import SegformerImageProcessor
from sklearn.model_selection import train_test_split
import re
import numpy as np


class CamvidDataset(Dataset):

  def __init__(self,
               root_dir,
               image_filenames,
               masks_filenames,
               feature_extractor,
               augment=False,
               num_classes=12) -> None:

    self.root_dir = root_dir
    self.image_filenames = image_filenames
    self.masks_filenames = masks_filenames
    self.num_classes = num_classes
    self.augment = augment
    self.feature_extractor = feature_extractor

    parent_dir = os.path.dirname(
      os.path.dirname(__file__)
    )
    if num_classes == 12:
      conf_file = os.path.join(parent_dir,'utils','label_colors11.txt')
    else:
      conf_file = os.path.join(parent_dir,'utils','label_colors.txt')
    
    colors, labels = self._dataset_conf(conf_file)
    self.id2label = dict(zip(range(self.num_classes),labels))
    self.class_colors = labelColors if num_classes == 12 else dict(zip(range(self.num_classes),colors))
  
  def __len__(self):
    return len(self.image_filenames)
  
  def __getitem__(self,idx):
    
    image_filename = self.image_filenames[idx]
    mask_filename = self.masks_filenames[idx]
    
    image = cv2.imread(os.path.join(self.root_dir,image_filename),)
    mask = cv2.imread(
            os.path.join(self.root_dir,mask_filename),
            cv2.IMREAD_UNCHANGED,
        )
    
    if self.num_classes != 12 :
      mask = self.bgr2gray(mask,self.class_colors)
    else:
      mask = self.bgr2gray12(mask,self.class_colors)
    
    if self.augment :
      image, mask = self._data_augmentation(image,mask)

    encod_inputs =self.feature_extractor(image,mask, return_tensors='pt')

    for k,v in encod_inputs.items():
      encod_inputs[k].squeeze_()

    return encod_inputs
  

  def bgr2gray(self,bgr_mask, colormap):

    mask_shape = bgr_mask.shape[:-1]+(self.num_classes,)
    mask = np.zeros(mask_shape)
    for label, rgb_color in colormap.items():
      bgr_color = rgb_color[::-1]
      mask[:,:,label] = np.all(bgr_mask == bgr_color,axis=-1)*1
    mask = np.argmax(mask,axis=-1)
    return mask
  

  def bgr2gray12(self,bgr_mask, colormap):

    mask_shape = bgr_mask.shape[:-1]+(self.num_classes,)
    mask = np.zeros(mask_shape)
    for idx, (label, rgb_colors) in enumerate(colormap.items()):
      for rgb_color in rgb_colors:
        bgr_color = rgb_color[::-1]
        mask[:,:,idx] += np.all(bgr_mask == bgr_color,axis=-1)*1
    mask = np.argmax(mask,axis=-1)
    return mask


  def _dataset_conf(self,filename = None):
    if not filename:
      filename = './utils/label_colors.txt'
    
    colors, class_names = [], []
    with open(filename) as f:
      lines = f.readlines()
    for line in lines:
      values = line.split()
      colors.append([int(value) for value in values[:3]])
      if not values[-1] in class_names :
        class_names.append(values[-1])
    return colors, class_names
  
  def _data_augmentation(self, image, mask):
    aug = A.Compose(
      [
          A.Flip(p=0.5),
          A.RandomRotate90(p=0.5),
          A.OneOf([
                  A.Affine(p=0.33,shear=(-5,5),rotate=(-80,90)),
                  A.ShiftScaleRotate(
                    shift_limit=0.2,
                    scale_limit=0.2,
                    rotate_limit=120,
                    p=0.33),
                  A.GridDistortion(p=0.33),
                ], p=1),
          A.CLAHE(p=0.8),
          A.OneOf(
              [
                  A.ColorJitter(p=0.33),
                  A.RandomBrightnessContrast(p=0.33),    
                  A.RandomGamma(p=0.33)
              ],
              p=1
          )
          ]
    )
    augmentation = aug(image=image, mask=mask)
    aug_img, aug_mask = augmentation['image'], augmentation['mask']
    return aug_img, aug_mask


def get_dataset(data_path='/dataset/camvid/',
                val_split=0.2,
                random_state=42,
                feature_extractor_name='nvidia/segformer-b2-finetuned-cityscapes-1024-1024'):
  
  feature_extractor = SegformerImageProcessor.from_pretrained(feature_extractor_name)
  feature_extractor.do_reduce_labels = False
  feature_extractor.do_resize = True
  feature_extractor.size = {"height":360, "width":480}
  feature_extractor.do_normalize= False
  feature_extractor.do_rescale= True


  img_files, mask_files = get_data_filenames(data_path)
  
  train_imgs, val_imgs, train_masks, val_masks = train_test_split(
      img_files, mask_files, test_size=val_split, random_state=random_state, shuffle=True)

  train_dataset = CamvidDataset(data_path,
                                train_imgs, train_masks,
                                feature_extractor,
                                augment=True,
                                num_classes=12
                                )
  val_dataset = CamvidDataset(data_path,
                              val_imgs, 
                              val_masks,
                              feature_extractor,
                              num_classes=12)
  return train_dataset, val_dataset

def get_data_filenames(data_path):
  files_list = list(iter(os.listdir(data_path)))
  img_regex = re.compile(r'00.*\d.png')
  gt_regex = re.compile(r'00.*_L.png')

  img_files = list(filter(img_regex.match, files_list))
  mask_files = list(filter(gt_regex.match, files_list))

  img_files.sort()
  mask_files.sort()

  return img_files, mask_files

def get_dataloader(dataset,
                   train_batch_size=10,
                   val_batch_size=7,
                   num_workers=2,
                   prefetch_factor=5):
  

  if isinstance(dataset,str):
    train_dataset, val_dataset = get_dataset(data_path=dataset)
  elif isinstance(dataset, list):
    train_dataset, val_dataset = dataset[0], dataset[1]
  
  else:
    raise TypeError(f"Expected dataset to be either string or list, got : {type(dataset)}")
    
  train_dataloader = DataLoader(train_dataset, 
                                batch_size=train_batch_size, 
                                shuffle=True, 
                                num_workers=num_workers, 
                                prefetch_factor=prefetch_factor)
  
  val_dataloader = DataLoader(val_dataset, 
                              batch_size=val_batch_size, 
                              num_workers=num_workers, 
                              prefetch_factor=prefetch_factor)
  
  return train_dataloader, val_dataloader
    

    
   
labelColors = {

"Sky":
    [
      [128, 128, 128] #; ... % "Sky"
    ]
,
"Building":
    [
      [0, 128 ,64], #; ... % "Bridge"
      [128, 0, 0], #; ... % "Building"
      [64, 192, 0], #; ... % "Wall"
      [64, 0, 64], #; ... % "Tunnel"
      [192, 0, 128] #; ... % "Archway"
    ]
,
"Pole":
    [
      [192, 192, 128], #; ... % "Column_Pole"
      [0 ,0 ,64] #; ... % "TrafficCone"
    ]
,
"Road":
    [
    [128 ,64 ,128], #; ... % "Road"
    [128 ,0, 192], #; ... % "LaneMkgsDriv"
    [192 ,0, 64], #; ... % "LaneMkgsNonDriv"
    ]
,
"Pavement":
    [
    [0, 0, 192], #; ... % "Sidewalk"
    [64, 192 ,128], #; ... % "ParkingBlock"
    [128 ,128 ,192], #; ... % "RoadShoulder"
    ]
,
"Tree":
    [
      [128, 128 ,0], #; ... % "Tree"
      [192, 192 ,0], #; ... % "VegetationMisc"
    ]
,
"SignSymbol":
    [
      [192, 128 ,128], #; ... % "SignSymbol"
      [128, 128 ,64], #; ... % "Misc_Text"
      [0, 64 ,64 ]#; ... % "TrafficLight"
    ]
,
"Fence":
    [
    [64, 64, 128] #; ... % "Fence"
    ]
,
"Car":
    [
    [64 ,0, 128], #; ... % "Car"
    [64 ,128, 192], #; ... % "SUVPickupTruck"
    [192 ,128, 192], #; ... % "Truck_Bus"
    [192, 64, 128], #; ... % "Train"
    [128, 64, 64] #; ... % "OtherMoving"
    ]
,
"Pedestrian":
    [
    [64, 64 ,0], #; ... % "Pedestrian"
    [192, 128 ,64], #; ... % "Child"
    [64, 0 ,192], #; ... % "CartLuggagePram"
    [64, 128, 64 ],#; ... % "Animal"
    ]
,
"Bicyclist":
    [
      [0, 128, 192], #; ... % "Bicyclist"
      [192 ,0, 192] #; ... % "MotorcycleScooter"
    ],

"Void":
    [
      [0, 0, 0], #; ... % "Void"
    ]
    }