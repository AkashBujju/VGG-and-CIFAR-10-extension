from PIL import Image
import numpy as np
import os


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


def dump_training_npy():
	all_directories = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck", "fruits"]
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
	np.save('all_images_training.npy', numpy_arr)
	print("shape: ", numpy_arr.shape)
	

def load_from_training_npy():
	np_arr = np.load("all_images_training.npy")
	test = np_arr[52000]

	assert_msg = 'Input shall be a HxWx3 ndarray'
	assert isinstance(test, np.ndarray), assert_msg
	assert len(test.shape) == 3, assert_msg
	assert test.shape[2] == 3, assert_msg

	img = Image.fromarray(test, "RGB")
	img.save("test.jpg")


if __name__ == "__main__":
	load_from_training_npy()
	#dump_training_npy()
