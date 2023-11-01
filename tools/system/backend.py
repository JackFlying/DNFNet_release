# coding: UTF-8
from flask import Flask, jsonify, request, send_file
from flask_cors import *
from gevent import pywsgi
from ps_model import load_model
from test_personsearch_prw_sys import get_prw_dataset_info, get_prw_data, search_performance_prw
from test_personsearch_cuhk_sys import get_cuhk_dataset_info, get_cuhk_data, search_performance_cuhk
from test_personsearch_my_sys import get_my_prw_dataset_info, get_input_prw_data, search_performance_input_prw
from vis_search import vis_search_result, vis_query
from utils import *
import numpy as np
import cv2
import base64
import torch
import os
import io
from base64 import encodebytes
from PIL import Image
from __init__ import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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

MY_PRW_Dataset, _, _ = get_my_prw_dataset_info(info_sota['PRW'])

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False  # 禁止中文转义
CORS(app,supports_credentials=True)

@app.route("/display_prw", methods=["POST", "GET"])
def display_prw():
    data_json = request.get_json()
    idx = data_json.get("idx")
    data = get_prw_data(PRW_Dataset, pname_to_attribute, idx)
    vis_query(data)
    return send_file(os.path.join(cache_dir, "query.png"), mimetype='image/jpeg')

@app.route("/display_cuhk", methods=["POST", "GET"])
def display_cuhk():
    data_json = request.get_json()
    idx = data_json.get("idx")
    data = get_cuhk_data(query_data_loader, idx)
    vis_query(data)
    return send_file(os.path.join(cache_dir, "query.png"), mimetype='image/jpeg')

@app.route("/upload_image", methods=["POST", "GET"])
def upload_image():
    try:
        data_json = request.get_json()
        encode_image = data_json.get("image_encoder")
        img = base64.b64decode(encode_image)
        image_data = np.fromstring(img, np.uint8)
        image_data = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        cv2.imwrite('./upload/query.jpg', image_data)
        return  jsonify({
                "code": 99999999,
                "msg": "文本为空"
            })
    except:
        return jsonify({
                "code": 99999999,
                "msg": "文本为空"
            })

def encoder_image(image_path):
    # img_stream = ''
    # with open(path, 'rb') as img_f:
    #     img_stream = img_f.read()
    #     img_stream = base64.b64encode(img_stream)
    # return img_stream

    pil_img = Image.open(image_path, mode='r') # reads the PIL image
    byte_arr = io.BytesIO()
    pil_img.save(byte_arr, format='PNG') # convert the PIL image to byte array
    encoded_img = encodebytes(byte_arr.getvalue()).decode('ascii') # encode as base64
    return encoded_img

def encoder_images():
    img_streams = {}
    for file in os.listdir(cache_dir):
        file_path = os.path.join(cache_dir, file)
        img_stream = encoder_image(file_path)
        img_streams[str(file)] = img_stream
    
    return jsonify({'result': img_streams})

def process_prw(data_json):
    idx = data_json.get("idx")
    model_type = data_json['model_type']
    data = get_prw_data(PRW_Dataset, pname_to_attribute, idx)
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
    data = get_cuhk_data(query_data_loader, idx)
    with torch.no_grad():
        if model_type == 1:
            result = model_cuhk_sota(return_loss=False, rescale=True, **data)
            entry = search_performance_cuhk(psdb_dataset, name_to_det_feat_cuhk_sota, result, idx, gallery_size=100)
        else:
            result = model_cuhk_base(return_loss=False, rescale=True, **data)
            entry = search_performance_cuhk(psdb_dataset, name_to_det_feat_cuhk_base, result, idx, gallery_size=100)
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
            entry = search_performance_input_prw(result, data, name_to_det_feat_prw_sota, gt_roidb)
        else:
            result = model_prw_base(return_loss=False, rescale=True, **data)
            entry = search_performance_input_prw(result, data, name_to_det_feat_prw_base, gt_roidb)
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