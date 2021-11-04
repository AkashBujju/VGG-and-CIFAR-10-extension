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

if __name__ == "__main__":
	# DO NOT RUN THE BELOW FUNCTION!!
	# dump_fruit_images_to_training_dir()

	all_directories = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck", "fruits"]
	train_dir = "cifar-10-images\\train\\"
	all_images = []

	for dir in all_directories:
		files = os.listdir(train_dir + dir)
		for file in files:
			full_path = train_dir + dir + "\\" + file
			im = Image.open(full_path)
			pix = im.load()
			current_image = []
			for i in range(0, 32):
				row_vals = []
				for j in range(0, 32):
					row_vals.append(pix[i, j])
				current_image.append(row_vals)
			all_images.append(current_image)
		print("Done loading ", dir)

	numpy_arr = np.array(all_images)
	np.save('all_images_training.npy', numpy_arr)
	print("shape: ", numpy_arr.shape)
