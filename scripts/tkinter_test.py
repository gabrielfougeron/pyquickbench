# import pyquickbench
import tkinter
import tkinter.filedialog
from PIL import ImageTk, Image
import os

window = tkinter.Tk()
# window.geometry("550x300+300+150")
window.resizable(width=True, height=True)

# def openfn():
#     filename = tkinter.filedialog.askopenfilename(title='open')
#     return filename
# 
# def open_img():
#     filename = openfn()
#     img = Image.open(filename)
#     img = img.resize((250, 250), Image.Resampling.LANCZOS)
#     img = ImageTk.PhotoImage(img)
#     panel = tkinter.Label(window, image=img)
#     panel.image = img
#     panel.pack()
# 
# btn = tkinter.Button(window, text='open image', command=open_img).pack()



for i in range(3):
    
    window.columnconfigure(i, weight=1, minsize=75)
    window.rowconfigure(i, weight=1, minsize=50)

    for j in range(0, 3):
        frame = tkinter.Frame(
            master=window,
            relief=tkinter.RAISED,
            borderwidth=1
        )
        frame.grid(row=i, column=j, padx=5, pady=5, sticky="nsew")
        label = tkinter.Label(master=frame, text=f"Row {i}\nColumn {j}")
        label.pack(padx=5, pady=5)


window.mainloop()