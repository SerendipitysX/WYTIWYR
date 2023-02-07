import sys
import os
current_path = os.path.dirname(os.getcwd())
print(current_path)
sys.path.append(current_path + '/utils')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from retrieval_utils import *
from bg_removel import bg_removal
from color_histogram import extract_color_hist
from PIL import Image
from flask import Flask, request
from flask_cors import CORS
import json
from arguments import args
app = Flask(__name__)
CORS(app)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================ Definition =====================================
type_list = ['Barchart', 'Histogram', 'Stacked Bar Chart', 'Box Plot', 'Circular Bar chart',
             'Scatter Chart', 'Pie Chart', 'Circular Packing Chart', 'Heatmap', 'Choropleth Map',
             'Line Chart', 'Dendrogram Chart', 'Network', 'Star Plot', 'Word Cloud', 'Sankey Diagram',
             'Timeline', 'Donut Chart']
trend_list = ['Increase Trend', 'Decrease Trend', 'Distribution']
layout_list = ['Horizontal Layout', 'Vertical Layout', 'Radial Layout']
color_list = ['Sequential Colormap', 'Diverging Colormap', 'Single Color', 'Categorical Colormap']
list_list = [type_list, trend_list, layout_list, color_list]
attribute_list = ['Type', 'Trend', 'Layout', 'Color']
requisite_attr = ['Type', 'Trend', 'Layout', 'Color']

transform = transforms.Compose([transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                    convert_image_to_rgb,
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                         (0.26862954, 0.26130258, 0.27577711)),
                                    ])


# ============================ dataset =====================================
classifier = pd.read_csv(current_path + '/data/annotation_18_30k.csv')
query_path = 'tem.png'
img_dir = current_path + '/data/all/'

@app.route('/retrieval', methods=['POST'])
def predict():
    # =================== Get user-oriented attributes ==========================
    # From Frontend!!!
    data_frontend = request.get_data()
    print(data_frontend)
    data_frontend = json.loads(data_frontend.decode('utf-8'))
    user_select_list = data_frontend['retrieval']
    user_inten_attr = data_frontend['new_clisifier']
    user_prompt = data_frontend['user']
    if user_prompt != []:
        user_prompt = data_frontend['user'][0]


    print(user_prompt)

    # reshape user select
    user_select = {}
    for attr in user_select_list:
        if attr in type_list:
            user_select['Type'] = attr
        if attr in trend_list:
            user_select['Trend'] = attr
        if attr in layout_list:
            user_select['Layout'] = attr
        if attr in color_list:
            user_select['Color'] = attr
    if user_inten_attr != []:
        user_select['CLIP'] = user_inten_attr[0]

    TREND_FLAG, COLOR_FLAG, USER_INTENT_FLAG = False, False, False
    if 'Trend' in user_select.keys():
        TREND_FLAG = True
    if 'Color' in user_select.keys():
        COLOR_FLAG = True
    if 'CLIP' in user_select.keys():
        USER_INTENT_FLAG = True

    # ==================== step1. filter imgs by attributes  =======================================
    print('------------------------------- Retrieve start! ----------------------------------------')
    # ==== step1.1 requisite attributes ===
    table_filtered = step1(user_select, table=annotation)
    print('After requisite attributes filtering, remain {} charts'.format(len(table_filtered)))
    # === step1.1 intent attributes ====
    if USER_INTENT_FLAG:
        # user_inten_attr_ = [i for i in user_inten_attr]
        user_inten_attr_ = read_list('user_prompt.json')
        print(user_inten_attr_)
        intent_attr_score = clip_attribute(user_select, table_filtered, user_inten_attr_, user_inten_attr)
        ### delete pred=0
        value_isnot_0_index = np.where(np.array(intent_attr_score) != 0)[0].astype(int).tolist()
        intent_attr_score_ = [intent_attr_score[i] for i in value_isnot_0_index]
        table_filtered = table_filtered.iloc[value_isnot_0_index]
        print('After intent attributes filtering, remain {} charts'.format(len(table_filtered)))


    # ==================== step2. Get global feature of filtered images & query  =======================================
    # === all ===
    all_ftrs = np.load(current_path + '/data/all_ftr.npy')
    all_ftrs = all_ftrs[table_filtered.index]
    # === query ===
    query_img = Image.open(query_path)
    query_ftr = get_feature_global(query_img).squeeze(0).cpu()


    # ==================== step3. Get similarity scores  =======================================
    # === Global perception ===
    score_global = similarity(all_ftrs, query_ftr)
    score_requisite = []
    # === CLIP ===
    if USER_INTENT_FLAG:
        print(len(intent_attr_score_), intent_attr_score_[0])
        score_requisite.append(list(map(lambda item: item * 1, intent_attr_score_)))
    # === Trend ===
    if TREND_FLAG:
        all_ftrs_gray = np.load(current_path + '/data/all_ftr_gray.npy')
        score_trend = similarity(all_ftrs_gray[table_filtered.index], query_ftr)
        score_requisite.append(score_trend)
    # === Color ===
    if COLOR_FLAG:
        query_color_hist = extract_color_hist(bg_removal(query_img))
        all_color_hist = np.load(current_path + '/data/all_ftr_color_hist.npy')
        score_color = similarity(all_color_hist[table_filtered.index], query_color_hist)
        # score_requisite.append(score_color)
        score_requisite.append(list(map(lambda item: item * 2, score_color)))
    if len(score_requisite) == 1:
        print('the score_requisite len is {}'.format(len(score_requisite)))
        score_requisite = score_requisite[0]
    if len(score_requisite) == 2:
        print('the score_requisite len is {}'.format(len(score_requisite)))
        score_requisite = list(map(sum, zip(score_requisite[0], score_requisite[1])))
    if len(score_requisite) == 3:
        print('the score_requisite len is {}'.format(len(score_requisite)))
        score_requisite = list(map(sum, zip(score_requisite[0], score_requisite[1], score_requisite[2])))


    # ==================== step4. sum or multiply the similarity =======================================
    if score_requisite == []:
        score_attr_final = score_global
    if score_requisite != []:
        score_attr_final = [score_global * np.exp(score_requisite)]
        score_attr_final = np.array(score_attr_final)
        score_attr_final = np.squeeze(score_attr_final)


    table_filtered = table_filtered.reset_index(drop=True)
    # ==================== step5. sum or multiply the similarity =======================================
    k = 5
    if user_prompt == []:
        have = ['file_name']
        if score_requisite != []:
            table_filtered['requisite_attr'] = score_requisite
            have.append('requisite_attr')
        if USER_INTENT_FLAG:
            table_filtered['intent_attr'] = intent_attr_score_
            have.append('intent_attr')
        table_filtered['global'] = score_global
        table_filtered['final_score'] = score_attr_final
        have += ['global', 'final_score']
        table_filtered = table_filtered.loc[:, have]
        table_filtered = table_filtered.sort_values(by=['final_score'], ascending=False)
        table_filtered.to_csv(current_path+'/data/table_filtered.csv', index=False)
        result_path = []
        for img_path in list(table_filtered['file_name'].values):
            is_same = check_same(query_path, img_dir, img_path)
            if not is_same:
                print(img_path)
                result_path.append(img_path)
            if len(result_path) == k:
                break
    else:
        multi_modality_ftr = get_feature_multi_modal(query_img, query_text=user_prompt)
        intent_prompt_score = np.abs(similarity(all_ftrs - query_ftr.cpu().numpy(), multi_modality_ftr.squeeze(0).cpu()))
        if score_requisite!= []:
            score = list(map(sum, zip(list(map(lambda item: item * args.mu, intent_prompt_score)), list(map(lambda item: item * args.nu, score_requisite)))))
        else:
            score = list(map(lambda item: item * args.mu, intent_prompt_score))
        score_attr_final = [score_global * np.exp(score)]
        score_attr_final = np.array(score_attr_final)
        score_attr_final = np.squeeze(score_attr_final)
        order_index = np.argsort(score_attr_final)[::-1]  # score descend
        have = ['file_name']
        if score_requisite != []:
            table_filtered['requisite_attr'] = score_requisite
            have.append('requisite_attr')
        table_filtered['prompt_attr'] = intent_prompt_score
        table_filtered['global'] = score_global
        table_filtered['score'] = score
        table_filtered['final_score'] = score_attr_final
        have += ['prompt_attr', 'global', 'score', 'final_score']
        table_filtered = table_filtered.loc[:, have]
        table_filtered = table_filtered.sort_values(by=['final_score'], ascending=False)
        table_filtered.to_csv(current_path + '/data/table_filtered.csv', index=False)
        result_path = []
        for img_path in list(table_filtered['file_name'].values):
            is_same = check_same(query_path, img_dir, img_path)
            if not is_same:
                print(img_path)
                result_path.append(img_path)
            if len(result_path) == k:
                break
    print('------------------------------- OK ----------------------------------------')
    return result_path

app.run(host=args.ip, port=args.port2, debug=True)