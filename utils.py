import torch
import torch.nn as nn
from torch.types import Device
import Config
from torchvision.utils import make_grid, save_image
import os
import cv2
import numpy as np

def gradient_penalty(critic, real, fake, image):
    _,c,h,w = real.shape;
    epsilon = torch.randint(0, 1, (Config.BATCH_SIZE,1,1,1)).repeat(1,c,h,w).to(Config.DEVICE);
    #epsilon.require_grad = True;
    epsilon = torch.tensor(epsilon, dtype=torch.float, requires_grad=True);
    interpolated_image = real*epsilon + fake * (1-epsilon);
    # interpolated_image_np = np.array(real.permute(0,2,3,1).cpu().detach().numpy(),np.uint8);
    # print(interpolated_image_np);
    # cv2.imshow("t",interpolated_image_np[0]*255);
    # cv2.waitKey();


    mixed_scores = critic(image, interpolated_image);
    gradient = torch.autograd.grad(
        inputs=interpolated_image,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True
    )[0];

    gradient = gradient.view(gradient.shape[0],-1);
    gradient_norm = gradient.norm(2,dim=1);
    gradient_penalty = torch.mean((gradient_norm - 1)**2);
    return gradient_penalty * 10.0;

def save_samples(model, val_loader, epoch, folder):
    x, y, _ = next(iter(val_loader))
    x, y = x.to(Config.DEVICE), y.to(Config.DEVICE)
    with torch.no_grad():
        y_fake,_ = model(x)
        y_fake = (torch.sigmoid(y_fake) > 0.5).long();
        
        b,_,_,_ = y_fake.size();

        fake_grid = make_grid(y_fake*255, b);
        save_image(fake_grid.float(), os.path.sep.join([folder, f"input_{epoch}.png"]))

        if epoch == 1:
            radiograph_grid = make_grid(x *0.5+ 0.5, b)
            save_image(radiograph_grid, os.path.sep.join([folder, f"radiograph.png"]))
            gt_grid = make_grid(y.float(), b)
            save_image(gt_grid, os.path.sep.join([folder, f"gt.png"]))


def save_checkpoint(model, epoch, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "epoch" : epoch
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, ):
    if(os.path.exists(checkpoint_file)):

        print("=> Loading checkpoint")
        
        checkpoint = torch.load(checkpoint_file, map_location=Config.DEVICE)
        check_dict = checkpoint['state_dict'];
        model_state = model.state_dict();
        for name, param in check_dict.items():
            if name in model_state and model_state[name].size() == check_dict[name].size():
                model_state[name].copy_(param);
            else:
                print(name);
        model.load_state_dict(model_state)
        
        #Return epoch number after successful loading
        return checkpoint['epoch'];
    return 0;