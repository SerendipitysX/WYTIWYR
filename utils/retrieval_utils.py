import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
import tqdm
from clip.model import CLIP
from clip.clip import tokenize
from transformers import CLIPProcessor, CLIPModel
from utils.dataloader import *
from classifier.backbone import get_model
from tqdm import tqdm
current_path = os.path.dirname(os.getcwd())

print(os.environ["CUDA_VISIBLE_DEVICES"])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
annotation = pd.read_csv(current_path + '/data/annotation_18_30k.csv')
img_dir = current_path + '/data/all/'


# ============================ Definition =====================================
type_list = ['Barchart', 'Histogram', 'Stacked Bar Chart', 'Box Plot', 'Circular Bar chart',
             'Scatter Chart', 'Pie Chart', 'Circular Packing Chart', 'Heatmap', 'Choropleth Map',
             'Line Chart', 'Dendrogram Chart', 'Network', 'Star Plot', 'Word Cloud', 'Sankey Diagram',
             'Timeline', 'Donut Chart']
trend_list = ['Increase Trend', 'Decrease Trend', 'Distribution']
layout_list = ['Horizontal Layout', 'Vertical Layout']
color_list = ['Sequential Colormap', 'Diverging Colormap', 'Single Color', 'Categorical Colormap']
list_list = [type_list, trend_list, layout_list, color_list]
attribute_list = ['Type', 'Trend', 'Layout', 'Color']
requisite_attr = ['Type', 'Trend', 'Layout', 'Color']


model_CLIP = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
model_CLIP.eval()
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def convert_image_to_rgb(image):
    return image.convert("RGB")

def load_image(img_origion):
    image = Image.open(img_origion)
    image = transform(image)
    return image

def read_json(file_name):
    with open(file_name) as handle:
        out = json.load(handle)
    return out

import pickle

# write list to binary file
def write_list(list_example, file_path):
    # store list in binary file so 'wb' mode
    with open(file_path, 'wb') as fp:
        pickle.dump(list_example, fp)
        print('Done writing list into a binary file')

# Read list to memory
def read_list(file_path):
    # for reading also binary mode is important
    with open(file_path, 'rb') as fp:
        n_list = pickle.load(fp)
        return n_list

def type_classifier():
    model_names = ["ResNet-50", "ResNet-50", "densenet-121", "densenet-201"]
    model_name = model_names[0]
    type_classifier_path = current_path + '/models/type_classifier.pth'
    model_type = get_model(model_name, 18).to(device)
    model_type.load_state_dict(torch.load(type_classifier_path))
    model_type.eval()
    return model_type


def trend_classifier():
    model_names = ["ResNet-50", "ResNet-50", "densenet-121", "densenet-201"]
    model_name = model_names[0]
    trend_classifier_path = current_path + '/models/trend_classifier.pth'
    model = get_model(model_name, 3).to(device)
    model.load_state_dict(torch.load(trend_classifier_path))
    model.eval()
    return model


def layout_classifier():
    model_names = ["ResNet-50", "ResNet-50", "densenet-121", "densenet-201"]
    model_name = model_names[0]
    layout_classifier_path = current_path + '/models/layout_classifier.pth'
    model = get_model(model_name, 2).to(device)
    model.load_state_dict(torch.load(layout_classifier_path))
    model.eval()
    return model


def color_classifier():
    model_names = ["ResNet-50", "ResNet-50", "densenet-121", "densenet-201"]
    model_name = model_names[0]
    color_classifier_path = current_path + '/models/color_classifier.pth'
    model = get_model(model_name, 4).to(device)
    model.load_state_dict(torch.load(color_classifier_path))
    model.eval()
    return model


def clip_classifier(img_path, attr):
    image = Image.open(img_dir + img_path)
    inputs = processor(text=[attr, 'others'], images=image, return_tensors="pt", padding=True).to(device)
    outputs = model_CLIP(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    return probs


# # =================== Extract feature from the queries ==========================
def get_feature_global(query, query_text=None):
    model_config_file = current_path + '/models/Vit-B-16.json'
    with open(model_config_file, 'r') as f:
        model_info = json.load(f)
    model = CLIP(**model_info).to(device)
    checkpoint = torch.load(current_path + '/models/Vit-B-16.pt')
    sd = checkpoint["state_dict"]
    if next(iter(sd.items()))[0].startswith('module'):
        sd = {k[len('module.'):]: v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model = model.cuda().eval()
    img1 = transform(query).unsqueeze(0).cuda()
    with torch.no_grad():
        sketch_feature = model.encode_image(img1)
        sketch_feature = sketch_feature / sketch_feature.norm(dim=-1, keepdim=True)
        if query_text != None:
            txt = tokenize([str(query_text)])[0].unsqueeze(0).cuda()
            text_feature = model.encode_text(txt)
            text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
            # return (sketch_feature+text_feature)/2
            return text_feature
    return sketch_feature


# # =================== Extract feature from the queries ==========================
def get_feature_multi_modal(query, query_text=None):
    model_config_file = current_path + '/models/Vit-B-16.json'
    with open(model_config_file, 'r') as f:
        model_info = json.load(f)
    model = CLIP(**model_info).to(device)
    checkpoint = torch.load(current_path + '/models/Vit-B-16.pt')
    sd = checkpoint["state_dict"]
    if next(iter(sd.items()))[0].startswith('module'):
        sd = {k[len('module.'):]: v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model = model.cuda().eval()
    img1 = transform(query).unsqueeze(0).cuda()
    with torch.no_grad():
        sketch_feature = model.encode_image(img1)
        sketch_feature = sketch_feature / sketch_feature.norm(dim=-1, keepdim=True)
        if query_text != None:
            txt = tokenize([str(query_text)])[0].unsqueeze(0).cuda()
            text_feature = model.encode_text(txt)
            text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
            # return (sketch_feature+text_feature)/2
            return sketch_feature - text_feature
    return sketch_feature


# # =================== Extractor requisite attribute for query ==========================
def get_feature(img, extractor):    # trend
    with torch.no_grad():
        feature = extractor(img)
        feature_norm = feature / feature.norm(dim=-1, keepdim=True)
        return feature_norm.cpu().numpy()


# =================== similarity cosine =========================
from numpy.linalg import norm

def similarity(A, B):
    cosine = np.dot(A, B)/(norm(A, axis=1)*norm(B))
    return cosine


# ============================= Step ==================================
def step1(user_select, table=annotation):
    '''
    return a table containing chart with all requisite attrs
    '''
    print('=== Step1 ===')
    for i in range(4):
        attr = requisite_attr[i]   # 'Type'
        if attr not in user_select.keys():
            continue
        else:
            attr_list = list_list[i]   # type_list
            label = attr_list.index(user_select[attr])
            if i in [0, 3]:   # type, color
                table = table[table[attr] == label]
            else:             # trend ,layout
                table = table[table[attr] == label]
    return table


def clip_attribute(user_select, table_filtered, user_inten_attr_, user_inten_attr):
    '''
    return each filtered chart's score in intent attribute
    '''
    with torch.no_grad():
        intent_attr_score = []
        img_paths = table_filtered.file_name.values
        bz = 32
        count = 0
        complete_bz = len(img_paths) // bz
        for index in tqdm(range(complete_bz)):
            imgs = []
            for i in range(bz):
                img_path = img_paths[count]
                image = Image.open(img_dir + img_path)
                imgs.append(image)
                count += 1
            inputs = processor(text=user_inten_attr_, images=imgs, return_tensors="pt", padding=True).to(device)
            outputs = model_CLIP(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
            index = torch.argmax(probs, dim=1).unsqueeze(1)
            probs = torch.gather(probs, 1, index)
            # torch.argmax(probs, dim=1).item()
            score = torch.where(index == user_inten_attr_.index(user_select['CLIP']), probs, 0)
            for score_value in score.cpu().numpy().squeeze():
                intent_attr_score.append(score_value)
        for index_ in range(count, len(img_paths)):
            image = Image.open(img_dir + img_paths[count])
            inputs = processor(text=user_inten_attr_, images=image, return_tensors="pt", padding=True).to(device)
            outputs = model_CLIP(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
            if torch.argmax(probs).item() == user_inten_attr_.index(user_select['CLIP']):
                intent_attr_score.append(torch.max(probs).cpu().item())
            else:
                intent_attr_score.append(0)
            count += 1
        print(len(intent_attr_score), len(img_paths))
        # assert len(intent_prompt_score) == len(img_paths)
    return intent_attr_score

def clip_prompt(table_filtered, intent_prompt):
    '''
    return each filtered chart's score in intent attribute
    '''
    with torch.no_grad():
        prompt = [intent_prompt, 'others']
        intent_prompt_score = []
        img_paths = table_filtered.file_name.values
        bz = 16
        count = 0
        complete_bz = len(img_paths) // bz
        for index in tqdm(range(complete_bz)):
            imgs = []
            for i in range(bz):
                img_path = img_paths[count]
                image = Image.open(img_dir + img_path)
                imgs.append(image)
                count += 1
            inputs = processor(text=prompt, images=imgs, return_tensors="pt", padding=True).to(device)
            outputs = model_CLIP(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
            index = torch.argmax(probs, dim=1).unsqueeze(1)
            probs = torch.gather(probs, 1, index)
            # torch.argmax(probs, dim=1).item()
            score = torch.where(index != prompt.index('others'), probs, 0)
            for score_value in score.cpu().numpy().squeeze():
                intent_prompt_score.append(score_value)
        for index_ in range(count, len(img_paths)):
            image = Image.open(img_dir + img_paths[count])
            inputs = processor(text=prompt, images=image, return_tensors="pt", padding=True).to(device)
            outputs = model_CLIP(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
            if torch.argmax(probs).item() == prompt.index('others'):
                intent_prompt_score.append(0)
            else:
                intent_prompt_score.append(torch.max(probs).cpu().item())
            count += 1
        print(len(intent_prompt_score), len(img_paths))
        # assert len(intent_prompt_score) == len(img_paths)
    return intent_prompt_score


import cv2
def check_same(query_path, img_dir, img_path):
    file1 = query_path
    file2 = img_dir + img_path
    img1 = cv2.imread(file1)
    img2 = cv2.imread(file2)
    try:
        difference = cv2.subtract(img1, img2)
        result = not np.any(difference)
        return result
    except:
        return False


def get_image_list(query_feat, nbrs, df):
    distances, indices = nbrs.kneighbors(query_feat.cpu().numpy())
    # distances, indices = nbrs.kneighbors(query_feat)
    im_list = []
    print(indices.shape, indices)
    for ind in indices[0]:
        file_loc = df.loc[ind, 'file_name']
        # file_loc = classifier.loc[ind, 'file_name']
        im_list.append(file_loc)
        print(file_loc)
    return im_list