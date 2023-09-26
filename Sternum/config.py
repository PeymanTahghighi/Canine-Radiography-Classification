import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

LEARNING_RATE = 7e-5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu';
BATCH_SIZE = 2;
IMAGE_SIZE = 512
EPSILON = 1e-5
WARMUP_EPOCHS = 20;
ENLARGE_RATE = 0.5



train_transforms_seg = A.Compose(
[
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    #A.PadIfNeeded(768,768,border_mode=cv2.BORDER_CONSTANT),
    #A.CropNonEmptyMaskIfExists(768,768),
    A.HorizontalFlip(p=0.5),
    A.CLAHE(clip_limit=2),
    A.Normalize(),
    ToTensorV2(),
],
additional_targets={'mask': 'mask'}
)


valid_transforms_seg = A.Compose(
    [
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.CLAHE(clip_limit=2,),
    A.Normalize(),
    ToTensorV2()
    ]
)