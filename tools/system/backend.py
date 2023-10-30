# coding: UTF-8
from flask import Flask, jsonify, request
from flask_cors import *
from gevent import pywsgi
from ps_model import load_model
from test_personsearch_prw_sys import get_prw_dataset_info, get_prw_data, search_performance_prw
from test_personsearch_cuhk_sys import get_cuhk_dataset_info, get_cuhk_data, search_performance_cuhk
from vis_search import vis_search_result
import torch
import os
from __init__ import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
info_sota = get_info_sota()

model_prw = load_model(info_sota['PRW'])
model_cuhk = load_model(info_sota['CUHK'])

PRW_Dataset, pname_to_attribute, gt_roidb, name_to_det_feat_prw = get_prw_dataset_info(info_sota['PRW'])
query_data_loader, psdb_dataset, name_to_det_feat_cuhk = get_cuhk_dataset_info(info_sota['CUHK'])

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False  # 禁止中文转义
CORS(app,supports_credentials=True)


# @app.route("/prw", methods=["POST", "GET"])
def process_prw():
    # data = request.get_json()
    # idx = data.get("idx", 2)
    idx = 2
    data = get_prw_data(PRW_Dataset, pname_to_attribute, idx)
    
    with torch.no_grad():
        result = model_prw(return_loss=False, rescale=True, **data)
    entry = search_performance_prw(result, data, pname_to_attribute, name_to_det_feat_prw, gt_roidb)
    
    vis_search_result(entry)
    # TODO post process

    # return jsonify({
    #     "code": 99999999,
    #     "msg": {
    #         'query':,
    #         'top-1':
    #     }
    # })

# @app.route("/cuhk", methods=["POST", "GET"])
def process_cuhk():
    # data = request.get_json()
    # idx = data.get("idx")
    # gallery_size = data.get("gallery_size", 100)
    
    idx = 15
    data = get_cuhk_data(query_data_loader, idx)
    with torch.no_grad():
        result = model_cuhk(return_loss=False, rescale=True, **data)
    entry = search_performance_cuhk(psdb_dataset, name_to_det_feat_cuhk, result, idx, gallery_size=100)

    vis_search_result(entry)
    # return jsonify({
    #     "code": 99999999,
    #     "msg": "文本为空"
    # })

if __name__ == '__main__':
    # server = pywsgi.WSGIServer(('0.0.0.0', 5001), app)
    # server.serve_forever()
    process_prw()
    process_cuhk()