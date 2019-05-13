import os
import tkinter as tk
from tkinter import Tk, Canvas, PanedWindow, Frame, Label, Entry
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

        # Main frame
        Label(self.frame, text='Theoritical d of Base:').grid(row=0, column=0, padx=10, pady=10)
        self.theoretical_d = Entry(self.frame, width=15)
        self.theoretical_d.grid(column=1, row=0)
        Label(self.frame, text='Å').grid(row=0, column=2)

        # Images of Base frame
        self.fm_base = Frame(pane, bd=2, relief='ridge', pady=20)
        pane.add(self.fm_base)
        Label(self.fm_base, text='○ Images of Base').grid(row=0, column=0)
        Label(self.fm_base, text='Images').grid(row=1, column=0, pady=100)
        Label(self.fm_base, text='Base Voltage').grid(row=2, column=0, pady=10)
        self.fm_base_img = Frame(self.fm_base, bd=2, relief='ridge')
        self.fm_base_img.grid(row=1, column=1, rowspan=2, sticky='W' + 'E' + 'N' + 'S', padx=35)
        ttk.Button(self.fm_base_img, text='select images',
                   command=self.select_file).grid(row=0, column=0, padx=500, pady=120)

        # Images of Molecules frame
        self.fm_molecules = Frame(pane, bd=2, relief='ridge', pady=20)
        pane.add(self.fm_molecules)
        Label(self.fm_molecules, text='○ Images of Molecules').grid(row=0, column=0)
        Label(self.fm_molecules, text='Images').grid(row=1, column=0, pady=100)
        Label(self.fm_molecules, text='Base Voltage').grid(row=2, column=0, pady=10)
        self.fm_molecules_img = Frame(self.fm_molecules, bd=2, relief='ridge', width=1000, height=200)
        self.fm_molecules_img.grid(row=1, column=1,
                                   rowspan=2, sticky='W' + 'E' + 'N' + 'S')
        ttk.Button(self.fm_molecules_img, text='select images',
                   command=self.select_file).grid(row=0, column=0, padx=500, pady=120)

        self.fm_run = Frame(pane, bd=2, relief='ridge', pady=10)
        pane.add(self.fm_run)
        ttk.Button(self.fm_run, text='RUN', command=self.run).pack()

    def run(self):
        pass

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
    app.master.geometry('1500x1000')
    app.master.title('Analyze LEED pattern')
    app.mainloop()


if __name__ == "__main__":
    main()
