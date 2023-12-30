# coding: UTF-8
from flask import Flask, jsonify, request
from flask_cors import *
from gevent import pywsgi
from ps_model import load_model
from test_personsearch_prw_sys import get_prw_dataset_info, get_prw_data, search_performance_prw
from test_personsearch_cuhk_sys import get_cuhk_dataset_info, get_cuhk_data, search_performance_cuhk
from test_personsearch_my_sys import get_my_prw_dataset_info, get_input_prw_data, search_performance_input_prw
from vis_search import vis_search_result, vis_query
import base64
import torch
import os
from __init__ import *

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
cache_dir = "/home/linhuadong/DNFNet/tools/system/cache"
info_sota = get_info_sota()
info_base = get_info_baseline()

model_prw_sota = load_model(info_sota['PRW'])
model_cuhk_sota = load_model(info_sota['CUHK'])
model_prw_base = load_model(info_base['PRW'])
model_cuhk_base = load_model(info_base['CUHK'])

PRW_Dataset, pname_to_attribute, gt_roidb, name_to_det_feat_prw_sota = get_prw_dataset_info(info_sota['PRW'])
_, _, _, name_to_det_feat_prw_base = get_prw_dataset_info(info_base['PRW'])
query_data_loader, psdb_dataset, name_to_det_feat_cuhk_sota = get_cuhk_dataset_info(info_sota['CUHK'])
_, _, name_to_det_feat_cuhk_base = get_cuhk_dataset_info(info_base['CUHK'])

_, _, name_to_det_feat_my_prw_sota = get_my_prw_dataset_info(info_sota['PRW'])
MY_PRW_Dataset, my_gt_roidb, name_to_det_feat_my_prw_base = get_my_prw_dataset_info(info_base['PRW'])

prw_idx_map = {1: 2, 2:3, 3:4, 4:6, 5:7, 6:9, 7:10, 8:16, 9:17, 10:18}
cuhk_idx_map = {1: 7, 2:11, 3:15, 4:20, 5:21, 6:22, 7:24, 8:31, 9:63, 10:69}

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False  # 禁止中文转义
CORS(app,supports_credentials=True)

@app.route("/display_prw", methods=["POST", "GET"])
def display_prw():
    data_json = request.get_json()
    idx = data_json.get("idx")
    data = get_prw_data(PRW_Dataset, pname_to_attribute, prw_idx_map[idx])
    vis_query(data)
    # return send_file(os.path.join(cache_dir, "query.png"), mimetype='image/jpeg')
    based64_image = encoder_image(os.path.join(cache_dir, "query.png"))
    return jsonify({
            "image":based64_image
        }
    )

@app.route("/display_cuhk", methods=["POST", "GET"])
def display_cuhk():
    data_json = request.get_json()
    idx = data_json.get("idx")
    data = get_cuhk_data(query_data_loader, cuhk_idx_map[idx])
    vis_query(data)
    based64_image = encoder_image(os.path.join(cache_dir, "query.png"))
    return jsonify({
            "image":based64_image
        }
    )
    # return send_file(os.path.join(cache_dir, "query.png"), mimetype='image/jpeg')

@app.route("/upload_image", methods=["POST", "GET"])
def upload_image():
    data_json = request.get_json()
    encode_image = data_json.get("image_encoder")
    head, context = encode_image.split(",")  # 将base64_str以“,”分割为两部分
    img = base64.b64decode(context)    # 解码时只要内容部分
    with open("./upload/query.jpg", 'wb') as f:
        f.write(img)
    return  jsonify({
            "code": 99999999,
            "msg": "文本为空"
        })

def encoder_image(image_path):
    img_stream = ''
    with open(image_path, 'rb') as img_f:
        img_stream = img_f.read()
        img_stream = base64.b64encode(img_stream).decode('utf-8')
    return img_stream

def encoder_images():
    img_streams = {}
    for file in os.listdir(cache_dir):
        file_path = os.path.join(cache_dir, file)
        img_stream = encoder_image(file_path)
        img_streams[str(file).split('.')[0]] = img_stream

    return jsonify(img_streams)

def process_prw(data_json):
    idx = data_json.get("idx")
    model_type = data_json['model_type']
    data = get_prw_data(PRW_Dataset, pname_to_attribute, prw_idx_map[idx])
    with torch.no_grad():
        if model_type == 1:
            result = model_prw_sota(return_loss=False, rescale=True, **data)
            entry = search_performance_prw(result, data, pname_to_attribute, name_to_det_feat_prw_sota, gt_roidb)
        else:
            result = model_prw_base(return_loss=False, rescale=True, **data)
            entry = search_performance_prw(result, data, pname_to_attribute, name_to_det_feat_prw_base, gt_roidb)
    vis_search_result(entry)
    return encoder_images()

def process_cuhk(data_json):
    idx = data_json.get("idx")
    model_type = data_json['model_type']
    data = get_cuhk_data(query_data_loader, cuhk_idx_map[idx])
    with torch.no_grad():
        if model_type == 1:
            result = model_cuhk_sota(return_loss=False, rescale=True, **data)
            entry = search_performance_cuhk(psdb_dataset, name_to_det_feat_cuhk_sota, result, cuhk_idx_map[idx], gallery_size=100)
        else:
            result = model_cuhk_base(return_loss=False, rescale=True, **data)
            entry = search_performance_cuhk(psdb_dataset, name_to_det_feat_cuhk_base, result, cuhk_idx_map[idx], gallery_size=100)
    vis_search_result(entry)
    return encoder_images()

def process_other(data_json):
    model_type = data_json['model_type']
    # image_name = data_json['image_name']
    # 保存图片
    data = get_input_prw_data(MY_PRW_Dataset, 'query.jpg')
    with torch.no_grad():
        if model_type == 1:
            result = model_prw_sota(return_loss=False, rescale=True, **data)
            entry = search_performance_input_prw(result, data, name_to_det_feat_my_prw_sota, my_gt_roidb)
        else:
            result = model_prw_base(return_loss=False, rescale=True, **data)
            entry = search_performance_input_prw(result, data, name_to_det_feat_my_prw_base, my_gt_roidb)
    vis_search_result(entry)  
    return encoder_images()
    
@app.route("/search", methods=["POST", "GET"])
def search():
    data = request.get_json()
    dataset_type = data['dataset_type']
    if dataset_type == 0:
        print("dataset_type == 0")
        return process_prw(data)
    elif dataset_type == 1:
        print("dataset_type == 1")
        return process_cuhk(data)
    elif dataset_type == 2:
        print("dataset_type == 2")
        return process_other(data)

if __name__ == '__main__':
    server = pywsgi.WSGIServer(('0.0.0.0', 5001), app)
    server.serve_forever()