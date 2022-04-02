import os
import uuid
import requests
from whitenoise import WhiteNoise
import time
import cv2

from flask import (Flask, flash, redirect, render_template, request,
                   send_from_directory, url_for)

from src import translate_dear, translate_vret

import numpy as np
import torch

from datetime import timedelta

YANDEX_API_KEY = 'YOUR API KEY HERE'
# SECRET_KEY = 'YOUR SECRET KEY FOR FLASK HERE'

app = Flask(__name__)
app.wsgi_app = WhiteNoise(app.wsgi_app, root='./static/')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(seconds=1)
# app.secret_key = SECRET_KEY


@app.route('/<path:path>')
def static_file(path):
    return app.send_static_file(path)


# force browser to hold no cache. Otherwise old result returns.
@app.after_request
def set_response_headers(response):
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

def save_video(file_path):
    import shutil
    base_path = r'G:\MSVD\YouTubeClips\mp4'
    target_path = r'E:\graduation\code\demo\VMT\static\videos'
    # 判断当前文件夹下面是否有，有就删除重新拷贝
    files = os.listdir(target_path)
    if files != 0:
        os.remove(os.path.join(target_path, files[0]))
    shutil.copyfile(os.path.join(base_path, file_path), os.path.join(target_path, file_path))

# main directory of programme
@app.route('/', methods=['GET', 'POST'])
def upload_file():

    try:
        # remove files created more than 5 minute ago
        os.system("find ./static/images/ -maxdepth 1 -mmin +5 -type f -delete")
    except OSError:
        pass

    if request.method == 'POST':

        # print(request.form['submit'])

        if request.form['submit'] == 'upload_video':
            video_name = request.files['content-file'].filename
            print(video_name)
            save_video(video_name)
            param = {'video': '../static/videos/' + video_name}
            print(param)
            return render_template('index.html', **param)

        elif request.form['submit'] == 'dear':
            src_sentence = request.form['src']
            vid = os.listdir(r"E:\graduation\code\demo\VMT\static\videos")[0]
            print(src_sentence)
            tgt_sentence = translate_dear.translate(vid.split('.')[0], src_sentence)
            param = {'dear_tgt': tgt_sentence,
                     'video': '../static/videos/' + vid,
                     'dear_src': src_sentence
            }
            print(param)
            return render_template('index.html', **param)

        elif request.form['submit'] == 'vret':
            src_sentence = request.form['src']
            vid = os.listdir(r"E:\graduation\code\demo\VMT\static\videos")[0]
            print(src_sentence)
            tgt_sentence = translate_vret.translate(vid.split('.')[0], src_sentence)
            param = {'vret_tgt': tgt_sentence,
                        'video': '../static/videos/' + vid,
                        'vret_src': src_sentence
            }
            print(param)
            return render_template('index.html', **param)

    return render_template('index.html')


@app.errorhandler(404)
def page_not_found(error):
    return render_template('page_not_found.html'), 404


if __name__ == "__main__":
    app.run(host='127.0.0.1', debug=False)
