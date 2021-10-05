from tkinter import *
from tkinter import filedialog
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import pyscreenshot as ImageGrab
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import lasagne
import theano
import theano.tensor as T
from PIL import Image
import dill as pickle
import cv2
from sklearn.utils import Bunch
from skimage.io import imread
from skimage.transform import resize
import joblib
import time

canvas_width = 550
canvas_height = 300

BG_GRAY = "#ABB2B9"
BG_COLOR = "#17202A"
BG_GRAY_DARK = "#A9A9A9"
TEXT_COLOR = "#EAECEE"
PRED_COLOR = "#FFFFFF"
FONT = "Helvetica 8 italic"
FONT_ALGO = "Helvetica 12"
FONT_BOLD = "Helvetica 13 bold"
OPTIONS = ["Digit Recognizer", "Fuit Recognizer"]


class GUI:

    def __init__(self):
        self.window = Tk()
        self._setup_main_window()

    def run(self):
        self.window.mainloop()

    def _setup_main_window(self):
        self.FILENAME = ""
        self.window.title("AI Recognition")
        self.window.resizable(width = False, height = False)
        self.window.configure(width = 470, height = 550, bg = BG_COLOR)

        head_label = Label(self.window, bg = BG_COLOR, fg = TEXT_COLOR, text = "Image Recognizer", font = FONT_BOLD, pady = 10)
        head_label.place(relwidth = 1)

        line = Label(self.window, width = 450, bg = BG_GRAY)
        line.place(relwidth = 1, rely = 0.07, relheight = 0.012)

        info_label = Label(self.window, bg = BG_COLOR, fg = TEXT_COLOR, text = "Select an image or draw it on the canvas", font = FONT)
        info_label.place(rely = 0.082)

        algo_label = Label(self.window, bg = BG_COLOR, fg = TEXT_COLOR, text = "Select type of recognizer: ", font = FONT_ALGO, pady = 10)
        algo_label.place(relx = 0.125, rely = 0.125)

        self.drop_val = StringVar(self.window)
        self.drop_val.set("Select an option")
        self.drop = OptionMenu(self.window, self.drop_val, *OPTIONS)
        self.drop.place(relx = 0.525, rely = 0.125)
        self.drop.configure(width = 15, bg = BG_COLOR, fg = TEXT_COLOR)

        self.canvas_widget = Canvas(self.window, width = 20, height = 2)
        self.canvas_widget.place(relheight = 0.54, relwidth = 1, rely = 0.2)
        self.canvas_widget.configure(cursor = "arrow")
        self.canvas_widget.bind("<B1-Motion>", self.paint)

        bottom_label = Label(self.window, bg = BG_GRAY_DARK, height = 80)
        bottom_label.place(relwidth = 1, rely = 0.825)

        browse_button = Button(self.window, text = "Browse", width = 20, font = FONT_BOLD, bg = BG_GRAY_DARK, command = self.browseFiles)
        browse_button.place(relx = 0.385, rely = 0.75, relwidth = 0.25, relheight = 0.04)

        send_button = Button(bottom_label, text = "Predict", width = 20, font = FONT_BOLD, bg = BG_GRAY_DARK, command = self.predict)
        send_button.place(rely = 0.001, relwidth = 1, relheight = 0.03)

        self.msg_entry = Entry(bottom_label, disabledbackground = "#17202A", fg = PRED_COLOR, font = FONT_BOLD)
        self.msg_entry.place(rely = 0.03, relheight = 0.04, relwidth = 1)
        self.msg_entry.configure(state = DISABLED)


    def predict(self):
        if self.drop_val.get() == "Digit Recognizer":
            x = self.window.winfo_rootx() + self.canvas_widget.winfo_x()
            y = self.window.winfo_rooty() + self.canvas_widget.winfo_y()
            x1 = x + self.canvas_widget.winfo_width()
            y1 = y + self.canvas_widget.winfo_height()
            image = np.asarray(ImageGrab.grab().crop((x, y, x1, y1)))

        if self.drop_val.get() == "Fuit Recognizer":
            if len(self.FILENAME) > 0:
                image = imread(self.FILENAME)
                image_resize = resize(image, (45, 45, 4), mode = 'reflect')
                image = np.array(image_resize.flatten())
                clf = joblib.load("fruitspredict.pkl")
                result = clf.predict([image])
                print(result)
                predicted_fruit = ""
                prep = "[]''"
                for char in result:
                    if char not in prep:
                        predicted_fruit = predicted_fruit + char
                predicted_fruit = "This is a " + predicted_fruit
                self.msg_entry.configure(state = NORMAL)
                self.msg_entry.delete(0, END)
                self.msg_entry.insert(END, predicted_fruit)
                self.msg_entry.configure(state = DISABLED)
            else:
                tk.messagebox.showerror(title = "No Image", message = "Image not selected")
        #time.sleep(15)
        #self.canvas_widget.delete("all")
        #self.msg_entry.configure(state = NORMAL)
        #self.msg_entry.delete(0, END)
        #self.msg_entry.configure(state = DISABLED)

    def paint(self, event):
       color = "#476042"
       x1, y1 = (event.x - 1), (event.y - 1)
       x2, y2 = (event.x + 1), (event.y + 1)
       self.canvas_widget.create_oval(x1, y1, x2, y2, fill = color)

    def clicked(self):
        pass

    def browseFiles(self):
        self.FILENAME = filedialog.askopenfilename(initialdir = "/home/jerinpaul/Pictures", title = "Select a File", filetypes =(("PNG", "*.png"),("JPG", "*.jpg"),("All Files","*.*")))
        print(self.FILENAME)
        if len(self.FILENAME):
            self.img = Image.open(self.FILENAME)
            self.img = self.img.resize((430, 240), Image.ANTIALIAS)
            self.img =  ImageTk.PhotoImage(self.img)
            self.canvas_widget.create_image(20,20, anchor = NW, image = self.img)
            self.canvas_widget.update()
        return

if __name__ == "__main__":
    app = GUI()
    app.run()

'''
def recognize_digit():
    x = win.winfo_rootx() + w.winfo_x()
    y = win.winfo_rooty() + w.winfo_y()
    x1 = x + w.winfo_width()
    y1 = y + w.winfo_height()
    image = np.asarray(ImageGrab.grab().crop((x,y,x1,y1)))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    _, binary = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)

    #contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    lowerBound = (80,120,100)
    upperBound = (160,200,180)
    thresh = cv2.inRange(image, lowerBound, upperBound)
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    for cntr in contours:
        print("Inside")
        x,y,wi,h = cv2.boundingRect(cntr)
        crop = image[y:y+h, x:x+wi]
        plt.imshow(crop)
        plt.show()
    #image = cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
    #image.reshape(-1, 1, 28, 28)
    #image / np.float32(256)

    #with open('model.dpkl', 'rb') as p_input:
    #    network = pickle.load(p_input)

    #with np.load('model.npz') as f:
    #    param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    #lasagne.layers.set_all_param_values(network, param_values)

    #test_prediction = lasagne.layers.get_output(network, deterministic = True)
    #val_fn = theano.function([input_var], test_prediction)
    #print(np.argmax(val_fn(image)))
    #plt.imshow(image)
    #plt.show()
    return

with open('model.dpkl', 'rb') as p_input:
    network = pickle.load(p_input)

with np.load('model.npz') as f:
    param_values = [f['arr_%d' % i] for i in range(len(f.files))]
lasagne.layers.set_all_param_values(network, param_values)
'''
