import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import torch.nn.functional as F
from torchvision.transforms.functional import normalize
import sys
import os
current_path = os.path.dirname(os.getcwd())
sys.path.append(current_path + '/utils')
from bg_removel_model import ISNetDIS
import warnings
warnings.filterwarnings("ignore")
current_path = os.path.dirname(os.getcwd())

resize = transforms.Resize(512)


def bg_removal(img):
    # model_path="./models/isnet-general-use.pth"  # the model path
    model_path= current_path + '/models/bg_removel.pth'  # the model path
    input_size=[1024,1024]
    net = ISNetDIS()
    print(torch.cuda.is_available())
    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_path))
        net = net.cuda()
    else:
        net.load_state_dict(torch.load(model_path, map_location="cpu"))
    net.eval()
    image_origin = img.convert("RGB")
    im = np.array(image_origin)
    if len(im.shape) < 3:
        im = im[:, :, np.newaxis]
    im_shp=im.shape[0:2]
    im_tensor = torch.tensor(im, dtype=torch.float32).permute(2, 0, 1)
    im_tensor = F.upsample(torch.unsqueeze(im_tensor,0), input_size, mode="bilinear").type(torch.uint8)
    image = torch.divide(im_tensor,255.0)
    image = normalize(image, [0.5,0.5,0.5], [1.0,1.0,1.0])
    image=image.cuda()
    result=net(image)
    result=torch.squeeze(F.upsample(result[0][0], im_shp,mode='bilinear'),0)
    ma = torch.max(result)
    mi = torch.min(result)
    result = (result-mi)/(ma-mi)
    a = (result[0] * 255).cpu().data.numpy().astype(np.uint8)
    avg = np.average(a)
    b = np.where(a > avg//2, 255, 0).astype(np.uint8)
    mask = Image.fromarray(b)
    image_origin.putalpha(mask)
    return image_origin