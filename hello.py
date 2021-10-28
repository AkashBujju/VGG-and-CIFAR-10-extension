# example of loading the cifar10 dataset
from matplotlib import pyplot
from tensorflow.keras.utils import to_categorical
from keras.datasets import cifar10
from keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from keras.layers import Dense, Activation, Flatten
from tensorflow.keras.optimizers import SGD

def load_dataset():
	(trainX, trainY), (testX, testY) = cifar10.load_data()
	trainY = to_categorical(trainY)
	testY = to_categorical(testY)
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
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(10, activation='softmax'))
	opt = SGD(learning_rate=0.001, momentum=0.9)
	print("about to compile")
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	print("compiled")
	return model

def run_test_harness():
	trainX, trainY, testX, testY = load_dataset()
	trainX, testX = prep_pixels(trainX, testX)
	model = define_model()
	print("About to fit model")
	history = model.fit(trainX, trainY, epochs=100, batch_size=256, validation_data=(testX, testY), verbose=1)
	print("Done fitting model")
	_, acc = model.evaluate(testX, testY, verbose=0)
	print("Done evaluating model")
	print('> %.3f' % (acc * 100.0))

if __name__ == "__main__":
	run_test_harness()
	print("DONE")
