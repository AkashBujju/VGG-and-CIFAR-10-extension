from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from tkinter import filedialog
from PIL import ImageTk, Image
import tkinter as tk
import numpy as np
 

def load_image(filename):
	img = load_img(filename, target_size=(32, 32))
	img = img_to_array(img)
	img = img.reshape(1, 32, 32, 3)
	img = img.astype('float32')
	img = img / 255.0
	return img


def run_example(filename):
	img = load_image(filename)
	model_filename = 'models\\model_with_fruits.h5'
	model = load_model(model_filename)
	print("Loaded model: ", model_filename)

	predict_x = model.predict(img)
	classes_x = np.argmax(predict_x, axis=1)
	names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck", "fruit", "human"]
	index = classes_x[0]
	return names[index]
 

class Demo:
	def __init__(self):
		self.window = tk.Tk()
		self.window.geometry("400x350")
		self.window.title("CIFAR-10")

		self.button = tk.Button(
		    text="Load Image",
		    width=15,
		    height=3,
			 command=self.load_new
		)
		self.button.pack()

		self.img = Image.open("test_images\\car_1.png")
		self.img = self.img.resize((200, 200), Image.ANTIALIAS)
		self.img = ImageTk.PhotoImage(self.img)
		self.label = tk.Label(
		    self.window,
		    image=self.img
		)
		self.label.pack(side=tk.BOTTOM)

		self.result_label = tk.Label(
			self.window,
			text="Automobile"
		)
		self.result_label.pack(side=tk.BOTTOM)


	def load_new(self):
		filename = filedialog.askopenfilename()
		self.img = Image.open(filename)
		self.img = self.img.resize((200, 200), Image.ANTIALIAS)
		self.img = ImageTk.PhotoImage(self.img)
		self.label.configure(image=self.img)

		result = run_example(filename)
		self.result_label.configure(text=result)


	def start_it(self):
		self.window.mainloop()


if __name__ == "__main__":
	demo = Demo()
	demo.start_it()
