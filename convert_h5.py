import coremltools
from keras.models import load_model

# modelをcoremlで使える形に変更
core_model = coremltools.converters.keras.convert(
	'animal_aug_cnn.h5',
	input_names='image', 
	image_input_names='image', 
	output_names='Prediction', 
	class_labels=['monkey', 'boar', 'crow'],
)

# 変換したmodelの保存
coreml_model.save('./animal_cnn_aug.mlmodel')