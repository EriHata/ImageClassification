import keras
from keras import Sequential, Input                            # NNのモデルを作る際に使用
from keras.layers import Conv2D, MaxPooling2D            # NNの部品
from keras.layers import Activation, Dropout, Flatten, Dense    # NNの部品　Denseは全結合層
from keras.utils import np_utils                                # 
import numpy as np
import warnings
warnings.simplefilter('ignore')

classes = ['monkey', 'boar', 'crow']
num_classes = len(classes)
image_size = 50

# 学習のモデルを定義する
def main():
	X_train, X_test, y_train, y_test = np.load('./animal.npy', allow_pickle=True)
	# 正規化
	X_train = X_train.astype('float') / 256  # 正規化
	X_test = X_test.astype('float') / 256    # 正規化
	# ラベルのone-hot-vector化
	y_train = np_utils.to_categorical(y_train, num_classes)
	y_test = np_utils.to_categorical(y_test, num_classes)

	# modelの学習
	model = model_train(X_train, y_train)
	# modelの評価
	model_eval(model, X_test, y_test)


def model_train(X, y):
	# modelの定義
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

	# なぜrmspropなんだろ
	opt = keras.optimizers.RMSprop(lr=0.0001, decay=1e-6)
	
	# loss, optimizer, metrics(評価手法)を決めてモデルを定義 
	model.compile(loss='categorical_crossentropy',
		optimizer=opt, metrics=['accuracy'])

	# 学習
	model.fit(X, y, batch_size=32, epochs=100)  # 処理が遅い場合はepoch数変えてみる

	# 学習したモデルの保存
	model.save('./animal_cnn.h5')

	# modelを返さないとmodel_evalでmodelを使ってテストができない
	return model


def model_eval(model, X, y):
	# modelの損失値と評価値を返す
	scores = model.evaluate(X, y, verbose=1)  # vervose:途中結果の表示
	print('Test Loss: ', scores[0])
	print('Test Accuracy: ', scores[1])

# このプログラムが直接pythonから呼ばれていたらmainを実行する
if __name__ == '__main__':
	main()
