from transformers import CLIPProcessor, CLIPModel
from utils.dataloader import *
from classifier.backbone import get_model
from flask import Flask, request
from flask_cors import CORS
from PIL import Image
current_path = os.path.dirname(os.getcwd())

app = Flask(__name__)
CORS(app)

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def _convert_image_to_rgb(image):
    return image.convert("RGB")

type_classifier_path = current_path + '/models/type_classifier.pth'
trend_classifier_path = current_path + '/models/trend_classifier.pth'
layout_classifier_path = current_path + '/models/layout_classifier.pth'
color_classifier_path = current_path + '/models/color_classifier.pth'
type_list = ['Barchart', 'Histogram', 'Stacked Bar Chart', 'Box Plot', 'Circular Bar chart',
             'Scatter Chart', 'Pie Chart', 'Circular Packing Chart', 'Heatmap', 'Choropleth Map',
             'Line Chart', 'Dendrogram Chart', 'Network', 'Star Plot', 'Word Cloud', 'Sankey Diagram',
             'Timeline', 'Donut Chart']
trend_list = ['Increase Trend', 'Decrease Trend', 'Distribution']
layout_list = ['Horizontal Layout', 'Vertical Layout', 'Radial Layout']
color_list = ['Sequential Colormap', 'Diverging Colormap', 'Single Color', 'Categorical Colormap']
list_list = [type_list, trend_list, layout_list, color_list]

classifier_dict = {'Barchart': [1, 2, 3], 'Histogram': [1, 2, 3], 'Stacked Bar Chart': [1, 2, 3],
                   'Box Plot': [2, 3], 'Circular Bar chart': [1, 2, 3], 'Scatter Chart': [1, 3],
                   'Pie Chart': [2, 3], 'Circular Packing Chart': [2, 3], 'Heatmap': [3], 'Choropleth Map': [3],
                   'Line Chart': [3], 'Dendrogram Chart': [2, 3], 'Network': [3], 'Star Plot': [2, 3], 'Word Cloud':[3],
                   'Sankey Diagram': [3], 'Timeline': [2, 3], 'Donut Chart': [2, 3]}
type_belong = {'Bar': ['Barchart', 'Stacked Bar Chart', 'Circular Bar chart'],
               'Circle': ['Pie Chart', 'Donut Chart', 'Circular Packing Chart'],
               'Diagram': ['Sankey Diagram', 'Timeline'],
               'Distribution': ['Histogram', 'Box Plot'],
               'Grid': ['Heatmap'],
               'Line': ['Line Chart', 'Star Plot'],
               'Map': ['Choropleth Map'],
               'Point': ['Scatter Chart'],
               'Text': ['Word Cloud'],
               'Tree': ['Dendrogram Chart', 'Network'],
               }

transform = transforms.Compose([transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                    _convert_image_to_rgb,
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                         (0.26862954, 0.26130258, 0.27577711)),
                                    ])

model_CLIP = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
model_CLIP.eval()
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def load_image(img_origion):
    image = Image.open(img_origion.stream)
    image = transform(image)
    return image


def get_key(value):
    for k, v in type_belong.items():
        if value in v:
            return k
        else:
            print('classifier.py file definition part should be rectified.')


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

@app.route('/attr_requisite', methods=['POST'])
def predict():
    file = request.files['file']
    tmp = Image.open(file.stream)
    tmp.save('tem.png', 'PNG')
    # session['file'] = file
    model_names = ["ResNet-50", "ResNet-50", "densenet-121", "densenet-201"]
    model_name = model_names[0]
    # type
    model_type = get_model(model_name, 18).to(device)
    model_type.load_state_dict(torch.load(type_classifier_path))
    model_type.eval()
    # trend
    model_trend = get_model(model_name, 3).to(device)
    model_trend.load_state_dict(torch.load(trend_classifier_path))
    model_trend.eval()
    # layout
    model_layout = get_model(model_name, 3).to(device)
    model_layout.load_state_dict(torch.load(layout_classifier_path))
    model_layout.eval()
    # color
    model_color = get_model(model_name, 4).to(device)
    model_color.load_state_dict(torch.load(color_classifier_path))
    model_color.eval()

    classifier_list = [model_type, model_trend, model_layout, model_color]
    attribute_list = ['Type', 'Trend', 'Layout', 'Color']

    img = load_image(file).unsqueeze(0)
    with torch.no_grad():
        img = img.to(device).to(torch.float32)
        y_pred_type = model_type(img)
        _, max_idx_type = torch.max(y_pred_type, 1)
        chart_type = type_list[max_idx_type]
        result_dict = {attribute_list[0]: chart_type}
        for i in classifier_dict[chart_type]:
            y_pred_attribute = classifier_list[i](img)
            _, max_idx_attribute = torch.max(y_pred_attribute, 1)
            chart_attribute = list_list[i][max_idx_attribute]
            result_dict[attribute_list[i]] = [chart_attribute]
        result_dict[attribute_list[0]] = [get_key(chart_type), chart_type]
    print('Annotation Ok', result_dict)
    return result_dict


@app.route('/attr_intent', methods=['POST'])
def predict2():
    print('------------------')
    texts = request.get_data()
    file = Image.open('tem.png')
    sentence = str(texts)
    first_kuohao = sentence.index('[')
    second_kuohao = sentence.index(']')
    texts_sentence = sentence[first_kuohao:second_kuohao + 1]
    texts = eval(texts_sentence)
    print(type(texts))
    print(texts)
    write_list(texts, './user_prompt.json')


    result_dict = {}
    if texts != []:
        inputs = processor(text=texts, images=file, return_tensors="pt", padding=True).to(device)
        outputs = model_CLIP(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        max_probs = torch.argmax(probs).item()
        result_dict['CLIP'] = texts[max_probs]
    print('Annotation Ok', result_dict)
    return result_dict

app.run(host='10.30.11.33', port=7779, debug=True)