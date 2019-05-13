import os
import tkinter as tk
from tkinter import Tk, Canvas, PanedWindow, Frame, Label, Button, Entry, Scrollbar
from tkinter import filedialog, ttk

import cv2
from PIL import Image, ImageTk


class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        pane = PanedWindow(self.master, orient='vertical')
        pane.pack(expand=True, fill=tk.BOTH, side='left')

        self.frame = Frame(pane, bd=2, relief='ridge')
        pane.add(self.frame)

        Label(self.frame, text='theoritical d of base:').grid(column=0, row=0, padx=10, pady=10)
        self.theoretical_d = Entry(self.frame, width=15)
        self.theoretical_d.grid(column=1, row=0)
        Label(self.frame, text='Å').grid(column=2, row=0)

        self.fm_base = Frame(pane, bd=2, relief='ridge', pady=10)
        pane.add(self.fm_base)
        Label(self.fm_base, text='○ images of base').grid(column=0, row=1)
        ttk.Button(self.fm_base, text='select images', command=self.select_file).grid(column=0, row=2)

        self.fm_molecules = Frame(pane, bd=2, relief='ridge', pady=10)
        pane.add(self.fm_molecules)
        Label(self.fm_molecules, text='○ images of molecules').grid(column=0, row=1)
        ttk.Button(self.fm_molecules, text='select images', command=self.select_file).grid(column=0, row=2)

    def get_d(self):
        print(self.theoretical_d.get())

    def select_file(self):
        initialdir = os.path.abspath(os.path.dirname("__file__"))
        files = filedialog.askopenfilenames(initialdir=initialdir)

        self.img_canvas = []
        self.image = []
        for i in range(len(files)):
            self.img_canvas.append(Canvas(self.fm_img))

        for i, filename in enumerate(files):
            self.draw_image(filename)
            self.img_canvas[i].create_image(0, 0, anchor='nw', image=self.image[i])
            self.img_canvas[i].pack()

    def draw_image(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, dsize=None, fx=0.3, fy=0.3)
        img = Image.fromarray(img)

        img = ImageTk.PhotoImage(img)
        self.image.append(img)


def main():
    root = Tk()
    app = Application(master=root)
    app.master.geometry('700x800')
    app.master.title('Analyze LEED pattern')
    app.mainloop()


if __name__ == "__main__":
    main()
