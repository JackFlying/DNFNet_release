# coding: UTF-8
from flask import Flask, jsonify, request
from flask_cors import *
from gevent import pywsgi
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False  # 禁止中文转义
CORS(app,supports_credentials=True)

#获取文本
@app.route("/wtext", methods=["POST","GET"])
def wtext():

    content=1
    if content :
        return jsonify({
            "code": 1,
        })
    else:
        return jsonify({
            "code": 99999999,
            "msg": "文本为空"
        })

if __name__ == '__main__':
    server = pywsgi.WSGIServer(('0.0.0.0', 5001), app)
    server.serve_forever()
