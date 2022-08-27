import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

LEARNING_RATE = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu';
BATCH_SIZE = 2;
IMAGE_SIZE = 1024
EPSILON = 1e-5

train_transforms = A.Compose(
[
    A.Resize(IMAGE_SIZE,IMAGE_SIZE),
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
    ToTensorV2(),
],
additional_targets={'mask': 'mask'}
)

valid_transforms = A.Compose(
    [
    A.Resize(IMAGE_SIZE,IMAGE_SIZE),
    A.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
    ToTensorV2()
    ]
)