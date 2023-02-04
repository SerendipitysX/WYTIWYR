import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image


def clip_predict(clip_model, processor, prompts, img_paths, device):
    all_probs = torch.zeros((len(img_paths), len(prompts)))
    for index, img_path in enumerate(img_paths):
        image_png = Image.open(img_path)
        outputs = clip_model(**processor(text=prompts, images=image_png, return_tensors="pt", padding=True).to(device))
        logits_per_image = outputs.logits_per_image
        all_probs[index] = logits_per_image.softmax(dim=1)
    return all_probs


class get_model(nn.Module):
    def __init__(self, name, class_num, m_path=None):
        super(get_model, self).__init__()
        if name == "ResNet-50":
            model = models.resnet50(pretrained=True)
            num_ftrs = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Linear(num_ftrs, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, class_num))
        if name == "ResNet-101":
            model = models.resnet101(pretrained=True)
            num_ftrs = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Linear(num_ftrs, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, class_num))
        if name == "densenet-121":
            model = models.densenet121(pretrained=True)
            num_ftrs = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Linear(num_ftrs, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, class_num))
        if name == "densenet-201":
            model = models.densenet201(pretrained=True)
            num_ftrs = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Linear(num_ftrs, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, class_num))
        self.model = model
        if m_path:
            checkpoint = torch.load(m_path)
            model.load_state_dict(checkpoint['model_state_dict'])


    def forward(self, images):
        y_pred = self.model(images)
        return y_pred
