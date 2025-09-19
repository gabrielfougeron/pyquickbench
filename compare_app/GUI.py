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
        
        self.width = 10 # Does not matter, will be changed automatically soon
        
        # TODO Change this for grid layout
        self.num_cols = self.master.rank_assign.k
        
        # self.img_dir = os.path.join(self.master.rank_assign.bench_root, "imgs")
        # self.all_images = []
        # for f in os.listdir(self.img_dir):
        #     root, ext = os.path.splitext(f)
        #     if ext.lower() in [".png", ".jpg"]:
        #         self.all_images.append(os.path.join(self.img_dir, f))

        self.scroll = ImageCompareScrollFrame(self, r=0, c=0, resize_images_func=self.resize_images).colcfg(range(1), weight=1).rowcfg(range(1), weight=1)
        self.scrollframe = self.scroll.frame

        self.all_lbls = []
        
        self.cache_init()
        self.fill_cache()
        self.resize_images()
        # self.display_current_choice()

    def on_key_press(self, event):

        ibest_choice = -2
        
        print(event.keysym)
        
        if event.keysym == "Escape":
            self.save_and_wrapup()
            
        elif event.keysym == "BackSpace":
            self.decr_cache()
            self.display_current_choice()
        
        if event.keysym == 'Left':
            if self.master.rank_assign.k == 2:
                ibest_choice = self.img_perm[0]
                
        elif event.keysym == 'Right':
            if self.master.rank_assign.k == 2:
                ibest_choice = self.img_perm[1]
                
        elif event.keysym == 'Up':
            ibest_choice = -1

        if ibest_choice >= -1:
            self.vote_current_choice(ibest_choice)
            self.incr_cache()
            self.display_current_choice()
            self.fill_cache()
            
    def save_and_wrapup(self):
        
        self.master.rank_assign.save_results()
        self.master.quit()
        
    def cache_init(self):
        
        self.n_prev_max = 3
        self.n_cache = self.n_prev_max + 2
        self.i_cache = 0
        
        self.img_perm = np.random.permutation(self.master.rank_assign.k)
        
        self.all_img_cache = []
        self.iset_cache = []
        self.vals_set_cache = []
        self.vote_cache = []
        
        self.fill_cache()
        
    def fill_cache(self):
        
        nfill = self.n_cache - len(self.all_img_cache)

        for ifill in range(nfill):

            iset, vals_set = self.master.rank_assign.next_set()

            width = self.width - self.num_cols * 4   # cater border and hightlight of labels
            img_width = width // self.num_cols
            
            img_cache = []
            tkimg_cache = []
            
            for val in vals_set:
                
                img_path = self.get_img_path(val)
                img = Image.open(img_path)
                ImageOps.exif_transpose(img, in_place=True)
                tkimg = ImageTk.PhotoImage(ImageOps.contain(img, (img_width, img_width)))
                
                img_cache.append(img)
                tkimg_cache.append(tkimg)
                
            self.iset_cache.append(iset)
            self.vals_set_cache.append(vals_set)
            self.all_img_cache.append([img_cache, tkimg_cache])
            self.vote_cache.append(None)

            
    def incr_cache(self):
        
        if self.i_cache < self.n_prev_max:
            self.i_cache += 1  
        
        else:

            self.iset_cache.pop(0)
            self.vals_set_cache.pop(0)
            img_cache, tkimg_cache = self.all_img_cache.pop(0)
            for img in img_cache:
                img.close()
            
    def decr_cache(self):
        
        if self.i_cache > 0:
            if self.vote_cache[self.i_cache] is not None:
                self.vote(self.i_cache, self.vote_cache[self.i_cache], mul = -1)
                self.vote_cache[self.i_cache] = None
            
            self.i_cache -= 1
            
    def vote_current_choice(self, ibest_choice):
        self.vote(self.i_cache, ibest_choice)
        
    def vote(self, i_cache, ibest_choice, mul = 1):
        
        self.vote_cache[i_cache] = ibest_choice
        self.master.rank_assign.vote_for_ibest(self.iset_cache[i_cache], ibest_choice, mul = mul)
        
    def display_current_choice(self, new_perm = True):
        
        for lbl in self.all_lbls:
            lbl.destroy()
        self.all_lbls = []
            
        cur_iset = self.iset_cache[self.i_cache]
        vals_set = self.vals_set_cache[self.i_cache]
        img_cache, tkimg_cache = self.all_img_cache[self.i_cache]

        assert len(vals_set) == self.master.rank_assign.k
        
        if new_perm:
            self.img_perm = np.random.permutation(self.master.rank_assign.k)
        
        for i in range(self.master.rank_assign.k):
            
            row, col = divmod(self.img_perm[i], self.num_cols)

            ### just create a label without showing the image initially
            lbl = ttk.Label(self.scrollframe, anchor="center")
            lbl.grid(row=row, column=col, sticky='nsew')
            lbl.configure(anchor="center", background='black')  
            lbl.config(image=tkimg_cache[i], anchor="center", background='black')
            
            self.all_lbls.append(lbl)
    
    def get_img_path(self, val):
        
        img_path = os.path.join(self.master.rank_assign.bench_root, "imgs", f"image_{str(int(val)).zfill(5)}_.png")
        
        return img_path

    def resize_images(self, width=None):
        
        if width is None:
            width = self.width
        else:
            self.width = width
        
        width -= self.num_cols * 4   # cater border and hightlight of labels
        ### calculate the grid size
        img_width = width // self.num_cols
        
        cache_wave = self.all_img_cache[self.i_cache]
        img_cache = cache_wave[0]
        tkimg_cache = []
        for img in img_cache:
            tkimg = ImageTk.PhotoImage(ImageOps.contain(img, (img_width, img_width)))
            tkimg_cache.append(tkimg)
        cache_wave[1] = tkimg_cache
        
        self.display_current_choice(new_perm = False)
        
        # ### go through all the labels
        # for lbl in self.scrollframe.winfo_children():
        #     ### resize the image keeping the aspect ratio
        #     img = ImageOps.contain(lbl.image, (img_width, img_width))
        #     ### show the image
        #     lbl.tkimg = ImageTk.PhotoImage(img)
        #     lbl.config(image=lbl.tkimg,anchor="center", background='black')

        for i_cache, cache_wave in enumerate(self.all_img_cache):
            
            if i_cache == self.i_cache:
                continue
            
            img_cache = cache_wave[0]
            
            tkimg_cache = []
            
            for img in img_cache:
                tkimg = ImageTk.PhotoImage(ImageOps.contain(img, (img_width, img_width)))
                tkimg_cache.append(tkimg)
                
            cache_wave[1] = tkimg_cache

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
