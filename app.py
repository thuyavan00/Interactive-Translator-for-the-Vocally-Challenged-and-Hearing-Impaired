
import tensorflow as tf
from flask import Flask, render_template, Response, request, flash, redirect, url_for, abort, send_file, send_from_directory
from sqlalchemy.ext.declarative import declarative_base
from werkzeug.utils import secure_filename
from db import db_init
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import create_engine
from sqlalchemy import Column, Integer, String, LargeBinary
from models import forumdb
import cv2
import numpy as np
import pyttsx3
import os
from base64 import b64encode
app=Flask(__name__, static_url_path='')
app.secret_key = "secret keey"
from tensorflow import keras
from db import db

camera = cv2.VideoCapture(0)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///img.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db_init(app)

model = keras.models.load_model(r'C:/Users/kanna/sign_final.h5')

IMG_WIDTH=200
IMG_HEIGHT=200
process_this_frame = True

classes = {'I': 0,
 'about': 1,
 'active': 2,
 'advice': 3,
 'after': 4,
 'afternoon': 5,
 'agree': 6,
 'always': 7,
 'ambulance': 8,
 'and': 9,
 'baby': 10,
 'call': 11,
 'can': 12,
 'cold': 13,
 'come': 14,
 'daddy': 15,
 'die': 16,
 'do': 17,
 'do not': 18,
 'flirt': 19,
 'fun': 20,
 'game': 21,
 'get': 22,
 'go': 23,
 'google': 24,
 'gun': 25,
 'hello': 26,
 'here': 27,
 'hot': 28,
 'life': 29,
 'nill': 30}
class_list = list(classes.keys())

mod_path = r'D:/sign/frozen_graph.pb'
weight_path = r'D:/sign/frozen_graph.pbtxt'
BASE_DIR = r"C:/Users/kanna/Desktop/fyp"
DOWNLOAD_DIRECTORY = r"C:/Users/kanna/Desktop/fyp/Files"
DOWNLOAD_DIRECTORY2 = r"C:/Users/kanna/Desktop/fyp/dist"
pr = [""]
def gen_frames():  
    while True:
        success, frame = camera.read()  
        if not success:
            break
        else:
            
            image=cv2.resize(frame, (IMG_HEIGHT, IMG_WIDTH),interpolation = cv2.INTER_AREA)
            image=np.array(image)
            image = image.astype('float32')
            image /= 255
            image.resize((1, 200, 200, 3))

            result = model.predict(image)
            p = np.argmax(result)
            
            
            class_name = class_list[p]
            
            engine = pyttsx3.init()
            engine.setProperty('voice', 'com.apple.speech.synthesis.voice.samantha')

            if class_name != 'nill' and pr[0]!=class_name:
                engine.say(str(class_name))
                pr[0] = class_name
                print(pr[0])
                engine.runAndWait()
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, 
                class_name, 
                (50, 120), 
                font, 1, 
                (0, 255, 255), 
                2, 
                cv2.LINE_4)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route("/forum",  methods=["GET", "POST"])
def forum():
    if request.method == "GET":
        return render_template("forum.html")
    name = request.form.get('name')
    title = request.form.get('title')
    text = request.form.get('text')
    pic = request.files.get('img')
    if not name:
        return 'No name', 400
    if not title:
        return 'No title', 400
    if not text:
        return 'No text', 400
    if not pic:
        return 'No pic uploaded!', 400

    filename = secure_filename(pic.filename)
    mimetype = pic.mimetype
    if not filename or not mimetype:
        return 'Bad upload!', 400
    q = forumdb(name=name, title=title, text = text, img = pic.read(), mimetype = mimetype)
    db.session.add(q)
    db.session.commit()

    return Response(flist())

@app.route("/flist", methods=["GET", "POST"])
def flist():
    res = db.session.query(forumdb.title, forumdb.id, forumdb.img, forumdb.name).all()
    #res = list(res)
    re = []
    i=0
    for r in res:
        #res[i] = list(r)
        re.append(list(r))
        re[i][2] = b64encode(r[2]).decode("utf-8")
        i+=1

    return render_template('flist.html', files=re)

@app.route("/train", defaults={'req_path': 'dist'})
@app.route('/<path:req_path>')
def train(req_path):
    abs_path = os.path.join(BASE_DIR, req_path)
    if not os.path.exists(abs_path):
        return abort(404)

    if os.path.isfile(abs_path):
        return send_file(abs_path)

    files = os.listdir(abs_path)

    return render_template('train.html', files=files)


@app.route("/serv_post", methods=["GET", "POST"])
def serv_post():
    title = request.form.get('title')
    id = request.form.get('id')
    res = db.session.query(forumdb.title, forumdb.id, forumdb.text, forumdb.name, forumdb.img).filter_by(id = id).all()
    image = b64encode(res[0][4]).decode("utf-8")
    return render_template('serv_post.html', files=res, image = image)

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/upload", methods=["GET", "POST"])
def upload():
    obj = request.files.get("file")
         
    ret_list = obj.filename.rsplit(".", maxsplit=1)
    if len(ret_list) != 2:
        return "Please upload valid file"
    if ret_list[1] != "zip":
        return "Please upload zip file"
 
    obj.save(os.path.join(BASE_DIR, "Files", obj.filename))
    flash('success')

    return Response(dir_listing('Files'))

@app.route('/download', defaults={'req_path': 'Files'})
@app.route('/<path:req_path>')
def dir_listing(req_path):
    abs_path = os.path.join(BASE_DIR, req_path)
    if not os.path.exists(abs_path):
        return abort(404)

    if os.path.isfile(abs_path):
        return send_file(abs_path)

    files = os.listdir(abs_path)

    return render_template('download.html', files=files)

@app.route('/serv',methods = ['GET','POST'])
def serv():
    result = request.form
    path = result['file']
    try:
        return send_from_directory(DOWNLOAD_DIRECTORY, path, as_attachment=True)
    except FileNotFoundError:
        abort(404)

@app.route('/servex',methods = ['GET','POST'])
def servex():
    result = request.form
    path = result['file']
    try:
        return send_from_directory(DOWNLOAD_DIRECTORY2, path, as_attachment=True)
    except FileNotFoundError:
        abort(404)

@app.route('/vfeed')
def vfeed():
    return render_template("vfeed.html")




@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=='__main__':
    app.run(debug=True)