import os
import tkinter
from tkinter import filedialog
from tkinter import ttk

import cv2
from PIL import Image, ImageTk


class Application(tkinter.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        pane = tkinter.PanedWindow(self.master, orient='horizontal')
        pane.pack(expand=True, fill=tkinter.BOTH, side="left")

        self.frame = tkinter.Frame(pane, bd=2, relief="ridge")
        pane.add(self.frame)

        tkinter.Label(self.frame, text='dの理論値：').grid(column=0, row=0, padx=10, pady=10)
        self.theoretical_d = tkinter.Entry(self.frame, width=15)
        self.theoretical_d.grid(column=1, row=0)
        tkinter.Label(self.frame, text='Å').grid(column=2, row=0)

        tkinter.Label(self.frame, text='○基盤の画像').grid(column=0, row=1)
        ttk.Button(self.frame, text='画像を選択', command=self.select_file).grid(column=0, row=2)

        fm_img = tkinter.Frame(pane, bd=2, relief="ridge")
        pane.add(fm_img)
        self.panel_img = tkinter.Label(fm_img)
        self.panel_img.pack()

    def get_d(self):
        print(self.theoretical_d.get())

    def select_file(self):
        initialdir = os.path.abspath(os.path.dirname("__file__"))
        files = filedialog.askopenfilenames(initialdir=initialdir)
        # files = filedialog.askopenfilename(filetypes=fTyp, initialdir=iDir)
        for filename in files:
            self.draw_image(filename)

    def draw_image(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, dsize=None, fx=0.3, fy=0.3)
        img_array = Image.fromarray(img)
        self.img = ImageTk.PhotoImage(img_array)
        self.panel_img.configure(image=self.img)
        self.panel_img.pack()


def main():
    root = tkinter.Tk()
    app = Application(master=root)
    app.master.geometry('600x800')
    app.mainloop()


if __name__ == "__main__":
    main()
