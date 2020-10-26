import os
from flask import Flask, request, redirect, url_for, flash
from werkzeug.utils import secure_filename  # ファイル名チェックする
from keras.models import load_model
from PIL import Image
import numpy as np

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTEMSIONS = set(['png', 'jpg', 'gif', 'jpeg'])

classes = ['monkey', 'boar', 'crow']
num_classes = len(classes)
image_size = 50

app = Flask(__name__)  # flaskのインスタンスを生成
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "super secret key"

# fileのアップロード可否判定関数
def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTEMSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
	# アップロードファイルがあれば保存する
	if request.method == 'POST':
		if 'file' not in request.files:
			flash('ファイルがありません')
			return redirect(request.url)  # fileをuploaするページに戻す
		file = request.files['file']      # fileがある場合はrequest.files['file']からfileのデータを取り出す
		if file.filename == '':
			flash('ファイルがありません')
			return redirect(request.url)  # fileをuploaするページに戻す
		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename)
			filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
			file.save(filepath)
			
			# fileを識別機に渡して結果を得る処理をここに書く
			model = load_model('./animal_aug_cnn.h5')
			image = Image.open(filepath)
			image = image.convert('RGB')  # gray scaleの画像が来たときもsyapeを統一
			image = image.resize((image_size, image_size))
			data = np.asarray(image)
			X = []
			X.append(data)
			X = np.array(X)  # pythonのリスト型からnumpyのndarryaへ
			result = model.predict([X])[0]
			predicted = result.argmax()                   # 最も大きい確率の添字を格納
			percentage = int(result[predicted]) * 100     # 確率をパーセンテージにする
			# print("{} ({} %)".format(classes[predicted], percentage)) web上に表示をしたいのでreturnで返す
			return 'ラベル：' + classes[predicted] + '　確率：' + str(percentage) + ' %'

			# return redirect(url_for('uploaded_file', filename=filename))   # upload後のページに転送
	return '''
	<!doctype html>
	<html>
	<head>
	<meta charset="UTF-8">
	<title>ファイルをアップロードして判定しよう</title></head>
	<body>
	<h1>ファイルアップロードして判定しよう！</h1>
	<form method = post enctype = multipart/form-data>
	<p><input type=file name=file>
	<input type=submit value=Upload>
	</form>
	</body>
	</html>
	'''

from flask import send_from_directory

@app.route('/uploads/<filename>')
def uploaded_file(filename):
	return send_from_directory(app.config['UPLOAD_FOLDER'], filename)