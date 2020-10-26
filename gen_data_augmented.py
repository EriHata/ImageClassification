from PIL import Image
import os, glob
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split

classes = ['monkey', 'boar', 'crow']
num_classes = len(classes)
image_size = 50
num_testdata = 100

# 画像読み込み
X_train = []
Y_train = []
X_test = []
Y_test = []

for index, classlabel in enumerate(classes):
	photos_dir = './' + classlabel
	files = glob.glob(photos_dir + '/*.jpg')  # globはパターン一致でファイル一致欄を取得
	for i, file in enumerate(files):
		if i >= 200:  # fileが200個を超えたらループを抜ける
			break
		image = Image.open(file)                         # pathを指定して画像読み込み
		image = image.convert('RGB')                     # 8bitのRGBモードに変換する
		image = image.resize((image_size, image_size))   # resize
		data = np.asarray(image)                         # ndarrayに変換
		
		if i < num_testdata:
			X_test.append(data)
			Y_test.append(index)
		else:
			# train dataの水増し
			for angle in range(-20, 20, 5):
				# 回転
				img_r = image.rotate(angle)
				data = np.asarray(img_r)
				X_train.append(data)
				Y_train.append(index)
			    # 左右反転
				img_t = img_r.transpose(Image.FLIP_LEFT_RIGHT)
				data = np.asarray(img_t)
				X_train.append(data)
				Y_train.append(index)



# リストをdnarrayに変換
X_train = np.array(X_train)
y_train = np.array(Y_train)
X_test = np.array(X_test)
y_test = np.array(Y_test)

# notebookでデータができているか確認

# X, Yをtrainとtestに分離する
#X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify=Y)
xy = (X_train, X_test, y_train, y_test)   # fileに一旦保存して後のプログラムから使えるようにする
np.save('./animal_aug.npy', xy)
# fileのサイズもだいたい10倍くらいになる
