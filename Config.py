import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

EPSILON = 1e-5
LEARNING_RATE = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu';
BATCH_SIZE = 1;
IMAGE_SIZE = 1400
D_PROJECT_ROOT = 'C:\\Users\\Admin\\OneDrive - University of Guelph\Miscellaneous\Diaphragm\\Labels'
ST_PROJECT_ROOT = 'C:\\Users\\Admin\\OneDrive - University of Guelph\Miscellaneous\\Sternum\\Labels'
SP_PROJECT_ROOT = 'C:\\Users\\Admin\\OneDrive - University of Guelph\Miscellaneous\\TipsSP\\Labels'
SR_PROJECT_ROOT = 'C:\\Users\\Admin\\OneDrive - University of Guelph\Miscellaneous\\Spine and Ribs\\Labels'
H_PROJECT_ROOT = 'C:\\Users\\Admin\\OneDrive - University of Guelph\Miscellaneous\\heart\\Labels'
IMAGE_DATASET_ROOT = 'C:\\Users\\Admin\\OneDrive - University of Guelph\Miscellaneous\\DVVD-Final'
SPINE_RIBS_LABEL_DIR = f'C:\\Users\\Admin\\OneDrive - University of Guelph\\Miscellaneous\\Spine and Ribs\\labels';
FULL_BODY_LABEL_DIR = 'C:\\Users\\Admin\\OneDrive - University of Guelph\\Miscellaneous\\Full radiograph segmentation\\labels';
ROOT_DATAFRAME_PATH = 'C:\\Users\\Admin\\OneDrive - University of Guelph\\Miscellaneous\\dvvd_list_final.xlsx';
CROP_SIZE = 1280;
WARMUP_EPOCHS = 10;
VIRTUAL_BATCH_SIZE = 4;
REBUILD_THORAX = False;
DEBUG_TRAIN_DATA = True;
RESUME = False;
TRAIN = False;

train_transforms = A.Compose(
[
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.CropNonEmptyMaskIfExists(CROP_SIZE, CROP_SIZE),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightness(),
    A.RandomContrast(),
    A.GaussNoise(),
    A.Normalize(),
    ToTensorV2(),
],
additional_targets={'mask': 'mask'}
)


valid_transforms = A.Compose(
    [
    A.Resize(CROP_SIZE, CROP_SIZE),
    A.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
    ToTensorV2()
    ],
additional_targets={'mask': 'mask'}
)

test_transforms = A.Compose(
    [
    A.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
    ToTensorV2()
    ],
additional_targets={'mask': 'mask'}
)