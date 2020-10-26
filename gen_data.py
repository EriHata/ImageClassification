from PIL import Image
import os, glob
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split

classes = ['monkey', 'boar', 'crow']
num_classes = len(classes)
image_size = 50

# 画像読み込み
X = []  # 画像データ
Y = []  # ラベル
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
		
		X.append(data)                                   # 画像データを格納
		Y.append(index)                                  # ラベル　0, 1, 2を格納

# リストをdnarrayに変換
X = np.array(X)
Y = np.array(Y)

# notebookでデータができているか確認

# X, Yをtrainとtestに分離する
X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify=Y)
xy = (X_train, X_test, y_train, y_test)   # fileに一旦保存して後のプログラムから使えるようにする
np.save('./animal.npy', xy)
