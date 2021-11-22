import os
from PIL import Image
import sys
import numpy as np
from matplotlib import pyplot
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dropout
from keras.layers import BatchNormalization


all_directories = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck", "fruits"]


def dump_fruit_images_to_testing_dir():
	fruits_testing_path = "external_images\\fruits\\testing\\"
	os.chdir(fruits_testing_path)
	all_dirs = [name for name in os.listdir(".") if os.path.isdir(name)]
	os.chdir("..\\..\\..\\")

	write_path = "cifar-10-images\\test\\fruits\\"
	des_file_names = ['{0:04}'.format(name) for name in range(0, 1100)]
	index = 0
	for dir in all_dirs:
		files = os.listdir(fruits_testing_path + dir + "\\")
		for i in range(0, 8):
			file = files[i]
			full_path = fruits_testing_path + dir + "\\" + file
			img = Image.open(full_path)
			img = img.resize((32, 32), Image.NEAREST)
			full_write_path = write_path + des_file_names[index] + ".jpg"
			index = index + 1
			img.save(full_write_path)
		print("Done saving ", dir)	

	return


def dump_fruit_images_to_training_dir():
	fruits_training_path = "external_images\\fruits\\training\\"
	os.chdir(fruits_training_path)
	all_dirs = [name for name in os.listdir(".") if os.path.isdir(name)]
	os.chdir("..\\..\\..\\")

	write_path = "cifar-10-images\\train\\fruits\\"
	des_file_names = ['{0:04}'.format(name) for name in range(0, 6000)]
	index = 0
	for dir in all_dirs:
		files = os.listdir(fruits_training_path + dir + "\\")
		for i in range(0, 39):
			file = files[i]
			full_path = fruits_training_path + dir + "\\" + file
			img = Image.open(full_path)
			img = img.resize((32, 32), Image.NEAREST)
			full_write_path = write_path + des_file_names[index] + ".jpg"
			index = index + 1
			img.save(full_write_path)

	return


def extend_testing_npy():
	train_dir = "external_images\\humans\\testing"
	numpy_array = np.load("all_images_testing.npy")

	dir_name = "external_images\\humans\\testing"
	files = os.listdir(dir_name)
	for file in files:
		full_path = dir_name + "\\" + file
		im = Image.open(full_path)
		im = im.convert('RGB')
		im = im.resize((32, 32), Image.NEAREST)
		im = np.asarray(im)
		im = im[np.newaxis, ...]
		numpy_array = np.append(numpy_array, im, 0)
	print("Done loading")

	print("type(numpy_array): ", type(numpy_array))
	print("shape(numpy_array): ", numpy_array.shape)

	numpy_array = numpy_array[:12000]
	np.save('all_images_testing.npy', numpy_array)
	print("saved...")

def extend_training_npy():
	train_dir = "external_images\\humans\\training"
	numpy_array = np.load("all_images_training.npy")

	dir_name = "external_images\\humans\\training"
	files = os.listdir(dir_name)
	for file in files:
		full_path = dir_name + "\\" + file
		im = Image.open(full_path)
		im = im.convert('RGB')
		im = im.resize((32, 32), Image.NEAREST)
		im = np.asarray(im)
		im = im[np.newaxis, ...]
		numpy_array = np.append(numpy_array, im, 0)
	print("Done loading")

	print("type(numpy_array): ", type(numpy_array))
	print("shape(numpy_array): ", numpy_array.shape)

	numpy_array = numpy_array[:60000]
	np.save('all_images_training.npy', numpy_array)
	print("saved...")


def dump_training_npy():
	train_dir = "cifar-10-images\\train\\"
	final_arr = []

	for dir in all_directories:
		files = os.listdir(train_dir + dir)
		for file in files:
			full_path = train_dir + dir + "\\" + file
			im = Image.open(full_path)
			im = np.asarray(im)
			final_arr.append(im)
		print("Done loading ", dir)

	numpy_arr = np.array(final_arr)
	numpy_arr = numpy_arr[:55000]
	np.save('all_images_training.npy', numpy_arr)
	print("shape: ", numpy_arr.shape)


def dump_testing_npy():
	test_dir = "cifar-10-images\\test\\"
	final_arr = []

	for dir in all_directories:
		files = os.listdir(test_dir + dir)
		for file in files:
			full_path = test_dir + dir + "\\" + file
			im = Image.open(full_path)
			im = np.asarray(im)
			final_arr.append(im)
		print("Done loading ", dir)

	numpy_arr = np.array(final_arr)
	numpy_arr = numpy_arr[0:11000]
	np.save('all_images_testing.npy', numpy_arr)
	print("shape: ", numpy_arr.shape)
	

def get_categorical():
	t1 = [[[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
	      [[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]],
	      [[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]],
	      [[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]],
	      [[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]],
	      [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]]

	trainY = []
	for i in range(0, 12):
		trainY.extend(t1[i].copy() * 5000)

	testY = []
	for i in range(0, 12):
		testY.extend(t1[i].copy() * 1000)
	
	return np.array(trainY), np.array(testY)


def load_dataset():
	trainX = np.load("all_images_training.npy")
	testX = np.load("all_images_testing.npy")
	trainY, testY = get_categorical()

	return trainX, trainY, testX, testY


def prep_pixels(train, test):
	train_norm = train.astype('float32')
	test_norm = test.astype('float32')
	train_norm = train_norm / 255.0
	test_norm = test_norm / 255.0
	return train_norm, test_norm


def define_model():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
	model.add(BatchNormalization())
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.2))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(BatchNormalization())
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.3))
	model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(BatchNormalization())
	model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.4))
	model.add(Flatten())
	model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
	model.add(BatchNormalization())
	model.add(Dropout(0.5))
	model.add(Dense(12, activation='softmax'))
	opt = SGD(learning_rate=0.001, momentum=0.9)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	return model


def summarize_diagnostics(history):
	pyplot.subplot(211)
	pyplot.title('Cross Entropy Loss')
	pyplot.plot(history.history['loss'], color='blue', label='train')
	pyplot.plot(history.history['val_loss'], color='orange', label='test')
	pyplot.subplot(212)
	pyplot.title('Classification Accuracy')
	pyplot.plot(history.history['accuracy'], color='blue', label='train')
	pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
	filename = sys.argv[0].split('/')[-1]
	pyplot.savefig(filename + '_plot2.png')
	pyplot.close()


def train_and_save_model():
	trainX, trainY, testX, testY = load_dataset()
	trainX, testX = prep_pixels(trainX, testX)
	model = define_model()
	datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
	it_train = datagen.flow(trainX, trainY, batch_size=1024)
	steps = int(trainX.shape[0] / 1024)
	history = model.fit(it_train, steps_per_epoch=steps, epochs=300, validation_data=(testX, testY), verbose=1)
	_, acc = model.evaluate(testX, testY, verbose=1)
	print('> %.3f' % (acc * 100.0))
	model.save('models\\model_with_fruits_and_humans.h5')


if __name__ == "__main__":
	train_and_save_model()
	print("Done.")
