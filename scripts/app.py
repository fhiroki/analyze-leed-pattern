import os
import tkinter as tk
from tkinter import Tk, Canvas, PanedWindow, Frame, Label, Entry, Scrollbar
from tkinter import filedialog, ttk

import cv2
from PIL import Image, ImageTk


class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.pack()
        self.create_widgets()

        self.base_image_paths = []
        self.base_images = []
        self.base_img_canvas = []
        self.base_voltages = []
        self.base_idx = 0

        self.mole_image_paths = []
        self.mole_images = []
        self.mole_img_canvas = []
        self.mole_voltages = []
        self.mole_idx = 0

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
        Label(self.fm_base, text='Base Voltage[V]').grid(row=2, column=0, pady=20)

        self.fm_base_canvas = Canvas(self.fm_base, width=1000)
        self.fm_base_canvas.grid(row=1, column=1, rowspan=2, sticky='news', padx=35)

        self.fm_base_bar = Scrollbar(self.fm_base, orient='horizontal', command=self.fm_base_canvas.xview)
        self.fm_base_bar.grid(row=3, column=1, sticky='ew', padx=35)
        self.fm_base_canvas.config(xscrollcommand=self.fm_base_bar.set)

        self.fm_base_canvas_frame = Frame(self.fm_base_canvas)
        self.fm_base_canvas.create_window((0, 20), window=self.fm_base_canvas_frame, anchor='nw')

        self.fm_base_btn = ttk.Button(self.fm_base, text='select images')
        self.fm_base_btn.config(command=lambda: self.select_file(isBase=True))
        self.fm_base_btn.grid(row=0, column=1, sticky='w', padx=30)

        # Images of Molecules frame
        self.fm_mole = Frame(pane, bd=2, relief='ridge', pady=20)
        pane.add(self.fm_mole)
        Label(self.fm_mole, text='○ Images of Molecules').grid(row=0, column=0)
        Label(self.fm_mole, text='Images').grid(row=1, column=0, pady=100)
        Label(self.fm_mole, text='Base Voltage[V]').grid(row=2, column=0, pady=20)

        self.fm_mole_canvas = Canvas(self.fm_mole, width=1000)
        self.fm_mole_canvas.grid(row=1, column=1, rowspan=2, sticky='news')

        self.fm_mole_bar = Scrollbar(self.fm_mole, orient='horizontal', command=self.fm_mole_canvas.xview)
        self.fm_mole_bar.grid(row=3, column=1, sticky='ew')
        self.fm_mole_canvas.config(xscrollcommand=self.fm_mole_bar.set)

        self.fm_mole_canvas_frame = Frame(self.fm_mole_canvas)
        self.fm_mole_canvas.create_window((0, 20), window=self.fm_mole_canvas_frame, anchor='nw')

        self.fm_mole_btn = ttk.Button(self.fm_mole, text='select images')
        self.fm_mole_btn.config(command=lambda: self.select_file(isBase=False))
        self.fm_mole_btn.grid(row=0, column=1, sticky='w', padx=30)

        # Run blob detection
        self.fm_run = Frame(pane, bd=2, relief='ridge', pady=10)
        pane.add(self.fm_run)
        ttk.Button(self.fm_run, text='RUN', command=self.run).pack()

    def run(self):
        print('Base')
        print('image: {}'.format(self.base_image_paths))
        print('voltages: {}\n'.format([base_voltage.get() for base_voltage in self.base_voltages]))
        print('Molecules')
        print('image: {}'.format(self.mole_image_paths))
        print('voltages: {}'.format([mole_voltage.get() for mole_voltage in self.mole_voltages]))

    def get_d(self):
        print(self.theoretical_d.get())

    def select_file(self, isBase):
        # initialdir = os.path.abspath(os.path.dirname("__file__"))
        initialdir = os.path.abspath(os.path.dirname('../data/'))
        files = filedialog.askopenfilenames(initialdir=initialdir)

        if isBase:
            for idx, filename in enumerate(files):
                i = self.base_idx
                self.base_img_canvas.append(Canvas(self.fm_base_canvas_frame))
                self.base_voltages.append(Entry(self.fm_base_canvas_frame))

                self.base_image_paths.append(filename)
                self.base_images.append(self.read_image(filename))
                self.base_img_canvas[i].create_image(self.base_images[i]['width'] / 2,
                                                     self.base_images[i]['height'] / 2,
                                                     image=self.base_images[i]['img'])
                self.base_img_canvas[i].grid(row=0, column=self.base_idx)
                self.base_voltages[i].grid(row=1, column=self.base_idx, pady=(20, 0))
                self.base_idx += 1

            self.fm_base_canvas_frame.update_idletasks()
            self.fm_base_canvas.config(scrollregion=self.fm_base_canvas.bbox('all'))
        else:
            for idx, filename in enumerate(files):
                i = self.mole_idx
                self.mole_img_canvas.append(Canvas(self.fm_mole_canvas_frame))
                self.mole_voltages.append(Entry(self.fm_mole_canvas_frame))

                self.mole_image_paths.append(filename)
                self.mole_images.append(self.read_image(filename))
                self.mole_img_canvas[i].create_image(self.mole_images[i]['width'] / 2,
                                                     self.mole_images[i]['height'] / 2,
                                                     image=self.mole_images[i]['img'])
                self.mole_img_canvas[i].grid(row=0, column=self.mole_idx)
                self.mole_voltages[i].grid(row=1, column=self.mole_idx, pady=(20, 0))
                self.mole_idx += 1

            self.fm_mole_canvas_frame.update_idletasks()
            self.fm_mole_canvas.config(scrollregion=self.fm_mole_canvas.bbox('all'))

    def read_image(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        height = int(img.shape[0] / 5)
        width = int(img.shape[1] / 5)
        img = cv2.resize(img, (width, height))
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(img)

        return {'img': img, 'height': height, 'width': width}


def main():
    root = Tk()
    app = Application(master=root)
    app.master.title('Analyze LEED pattern')
    app.mainloop()


if __name__ == "__main__":
    main()
