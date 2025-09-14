import os
import numpy as np
import tkinter as tk, tkinter.ttk as ttk
from typing import Iterable
from PIL import Image, ImageTk, ImageOps
import pyquickbench

class ImageCompareScrollFrame(tk.Frame):
    
    def __init__(self, master, r=0, c=0, **kwargs):
        ### get the resize callback
        self.resize_images_func = kwargs.pop("resize_images_func", None)
        tk.Frame.__init__(self, master, **{'width': 400, 'height': 300, **kwargs})

        self.master.grid(sticky = 'nswe')
        self.grid(sticky = 'nswe')

        # give this widget weight on the master grid
        self.master.grid_rowconfigure(r, weight=1)
        self.master.grid_columnconfigure(c, weight=1)

        # give self.frame weight on this grid
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # _WIDGETS
        self.canvas = tk.Canvas(self, bd=0, highlightthickness=0)
        self.canvas.configure(background='black')
        self.canvas.grid(row=0, column=0, sticky='nswe')

        self.frame = tk.Frame(self.canvas, **kwargs)
        # self.frame = tk.Frame(self.canvas, anchor='center', **kwargs)
        self.frame_id = self.canvas.create_window((0, 0), window=self.frame, anchor="center")

        self.canvas.bind("<Configure>", self.on_canvas_configure)
        self.frame.bind("<Configure>", self.on_frame_configure)

        self.task = None   

    # makes frame width match canvas width
    def on_canvas_configure(self, event):
        # function to do the final resize task
        def final_update(width):
            
            self.canvas.itemconfig(self.frame_id, width=width)
            if self.resize_images_func:
                self.resize_images_func(width)
            self.task = None

        if self.task:
            self.after_cancel(self.task)
        # don't do resize task during resizing the canvas to improve performance
        # so delay the task using .after()
        self.task = self.after(50, final_update, event.width)

    # when frame dimensions change pass the area to the canvas scroll region
    def on_frame_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        self.canvas.configure()

    # configure self.frame row(s)
    def rowcfg(self, index, **options):
        index = index if isinstance(index, Iterable) else [index]
        for i in index:
            self.frame.grid_rowconfigure(i, **options)
        # so this can be used inline
        return self

    # configure self.frame column(s)
    def colcfg(self, index, **options):
        index = index if isinstance(index, Iterable) else [index]
        for i in index:
            self.frame.grid_columnconfigure(i, **options)
        # so this can be used inline
        return self


class ImageCompareAuxiliaryWindow(tk.Frame):
    
    def __init__(self, master, **kwargs):
        tk.Frame.__init__(self, master, **kwargs)
        
        self.img_dir = os.path.join(self.master.rank_assign.bench_root, "imgs")
        self.all_images = []
        for f in os.listdir(self.img_dir):
            root, ext = os.path.splitext(f)
            if ext.lower() in [".png", ".jpg"]:
                self.all_images.append(os.path.join(self.img_dir, f))

        self.scroll = ImageCompareScrollFrame(self, r=0, c=0, resize_images_func=self.resize_images).colcfg(range(1), weight=1).rowcfg(range(1), weight=1)
        self.scrollframe = self.scroll.frame

        self.fillScrollRegion()

    def on_key_press(self, event):

        ibest_choice = -1
        
        print(f'{event.keysym = }')
        
        if event.keysym == "Escape":
            self.save_and_wrapup()
        
        if event.keysym == 'Left':
            if self.master.rank_assign.k == 2:
                ibest_choice = 0
        elif event.keysym == 'Right':
            if self.master.rank_assign.k == 2:
                ibest_choice = 1

        if ibest_choice >= 0:
            # self.master.rank_assign.vote_for_ibest(ibest_choice)
            self.fillScrollRegion()
            self.resize_images()
    
    def save_and_wrapup(self):
        
        self.master.rank_assign.save_results()
        self.master.quit()
    
    def fillScrollRegion(self):

        n_images = len(self.all_images)
        
        ### use instance variable to store the number of columns
        self.num_cols = self.master.rank_assign.k
        for i in range(self.num_cols):
            
            row, col = divmod(i, self.num_cols)
            i_img = np.random.randint(n_images)
            
            img = Image.open(self.all_images[i_img])
            img = ImageOps.exif_transpose(img)
            ### just create a label without showing the image initially
            lbl = ttk.Label(self.scrollframe, anchor="center")
            lbl.grid(row=row, column=col, sticky='nsew')
            lbl.configure(anchor="center", background='black')  
            lbl.image = img  ### save the original image for later resize

    def resize_images(self, width=None):
        
        if width is None:
            width = self.width
        else:
            self.width = width
        
        width -= self.num_cols * 4   # cater border and hightlight of labels
        ### calculate the grid size
        img_width = width // self.num_cols
        ### go through all the labels
        for lbl in self.scrollframe.winfo_children():
            ### resize the image keeping the aspect ratio
            img = ImageOps.contain(lbl.image, (img_width, img_width))
            ### show the image
            lbl.tkimg = ImageTk.PhotoImage(img)
            lbl.config(image=lbl.tkimg,anchor="center", background='black')

    def do(self):
        pass

class ImageCompareGUI(tk.Tk):
    
    def __init__(self, rank_assign):
        tk.Tk.__init__(self)
        
        self.title('Ultra Minimalist GUI')
        self.rank_assign = rank_assign

        aux = ImageCompareAuxiliaryWindow(self)
        aux.pack(expand=1, fill="both")
        
        self.bind("<KeyPress>", aux.on_key_press)

    def __call__(self):
        self.mainloop()
