# evaluate the deep model on the test dataset
from keras.datasets import cifar10
from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical
import numpy as np
 
def load_image(filename):
	img = load_img(filename, target_size=(32, 32))
	img = img_to_array(img)
	img = img.reshape(1, 32, 32, 3)
	img = img.astype('float32')
	img = img / 255.0
	return img

def run_example():
	img = load_image('test_images\\orange_1.jpg')
	model = load_model('models\\custom_model.h5')
	predict_x = model.predict(img)
	classes_x = np.argmax(predict_x, axis=1)
	print("result: ", classes_x[0])
 

if __name__ == "__main__":
	run_example()
