import keras
from keras import Sequential, Input
from keras.models import load_model                          # NNのモデルを作る際に使用
from keras.layers import Conv2D, MaxPooling2D            # NNの部品
from keras.layers import Activation, Dropout, Flatten, Dense    # NNの部品　Denseは全結合層
from keras.utils import np_utils                                # 
import numpy as np
from PIL import Image
import sys
import warnings
warnings.simplefilter('ignore')

classes = ['monkey', 'boar', 'crow']
num_classes = len(classes)
image_size = 50


def build_model():
	# modelの定義
	"""
	model = Sequential()
	model.add(Input(shape=(50, 50, 3)))
	model.add(Conv2D(32, 3, padding='same', activation="relu"))
	model.add(Conv2D(32, 3, activation="relu"))
	model.add(MaxPooling2D(2))
	model.add(Dropout(0.25))

	model.add(Conv2D(64, 3, padding='same', activation="relu"))
	model.add(Conv2D(64, 3, padding='same', activation="relu"))
	model.add(MaxPooling2D(2))
	model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(512, activation="relu"))
	model.add(Dropout(0.5))
	model.add(Dense(3, "softmax"))


	opt = keras.optimizers.RMSprop(lr=0.0001, decay=1e-6)
	
	# loss, optimizer, metrics(評価手法)を決めてモデルを定義 
	model.compile(loss='categorical_crossentropy',
		optimizer=opt, metrics=['accuracy'])

	# 学習
	# model.fit(X, y, batch_size=32, epochs=30)  # 処理が遅い場合はepoch数変えてみる
	"""

	# すでに保存したモデルのロード
	# これはパラメータを格納しているのか？
	model = load_model('./animal_aug_cnn.h5')

	# modelを返さないとmodel_evalでmodelを使ってテストができない
	return model


def main():
	# 入力画像の取得と前処理
	image = Image.open(sys.argv[1])
	image = image.convert('RGB')  # gray scaleの画像が来たときもsyapeを統一
	image = image.resize((image_size, image_size))
	data = np.asarray(image)
	X = []
	X.append(data)
	X = np.array(X)
	model = build_model()

	result = model.predict([X])[0]
	predicted = result.argmax()                   # 最も大きい確率の添字を格納
	percentage = int(result[predicted]) * 100     # 確率をパーセンテージにする
	print("{} ({} %)".format(classes[predicted], percentage))


if __name__ == "__main__":
	main()
