# -*- coding: utf-8 -*-
"""
Created on 15th feb 2020

@author: Adminz,gagan.
"""
import numpy as np
import argparse
import cv2 as cv
import subprocess
import time
import os
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow
import tkinter as tk
from alphabet_recognition import live_feed

global alphabet, sign, searched
LARGE_FONT= ("Verdana", 12)
val=0

class ObjDet(tk.Tk):
    def __init__(self, *args, **kwargs):
        
        tk.Tk.__init__(self, *args, **kwargs)
        container = tk.Frame(self)
        self.title("Sign In Air")
        self.configure(bg='black')
        container.pack()

        self.frames = {}
        self.geometry("800x900")

        for F in (StartPage, livefeed):
            frame = F(container, self)
            self.frames[F] = frame
            frame.grid(row=900, column=800)
        self.show_frame(StartPage)

    def show_frame(self, cont):

        frame = self.frames[cont]
        frame.tkraise()

from tkinter.filedialog import askopenfilename
import pickle

class StartPage(tk.Frame): #initial page of the GUI
    def __init__(self, parent, controller):
        tk.Frame.__init__(self,parent)
        
        
        self.image1 = tk.PhotoImage(file='signinair.png')
        
        panel1 = tk.Label(self, image=self.image1)
        panel1.place(x=35, y=200)
        
        panel1.image = self.image1
        
        label = tk.Label(self, text="Detection of Sign In AIR", font=LARGE_FONT)
        label2 = tk.Label(self, text="To distinguish different Sign", font=LARGE_FONT)
        label.pack(pady=20,padx=20,side="top", fill="both", expand = False)
        label2.pack()
        
        tk.Label(self, text="Alphabet", font="Verdana 10").pack(pady=15,padx=5)

        button = tk.Button(self, text="Detection on Video",command=lambda: controller.show_frame(livefeed))
        button.pack(pady=530,padx=110,side="right")
        
        button1 = tk.Button(self, text="Detection on Image",command=lambda: controller.show_frame(livefeed)).place(x=290, y=670)

        button2 = tk.Button(self, text="Detection on Live Feed",command=lambda: controller.show_frame(livefeed))
        button2.pack(pady=530,padx=110,side="left")
        
class livefeed(tk.Frame):
    def __init__(self, parent, controller):
        val=2
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Detection of Sign In AIR", font=LARGE_FONT)
        label.pack(pady=50,padx=50)
        
        tk.Label(self, text = "Select the Alphabet or Sign", font="Times 14").place(x=210, y=80)
        
        alphabet = tk.IntVar()
        sign = tk.IntVar()
        tk.Checkbutton(self, text="Alphabet", variable=alphabet).place(x=310, y=110)
        tk.Checkbutton(self, text="Sign", variable=sign).place(x=310, y=140)
        
        tk.Button(self, text ="Detect", command = lambda: check_class(alphabet,sign,val)).place(x=300, y=240)

        button = tk.Button(self, text="Home",
                            command=lambda: controller.show_frame(StartPage))
        button.pack(pady=530,padx=130,side="left")
    
        button1 = tk.Button(self, text="Detection on Image",command=lambda: controller.show_frame(livefeed)).place(x=250, y=654)

        button2 = tk.Button(self, text="Detection on Live Feed",command=lambda: controller.show_frame(livefeed))
        button2.pack(pady=530,padx=110,side="left")
        

def DisplayDir(Var):
    feedback = askopenfilename()
    Var.set(feedback)

def check_class(alphabet,sign,val,searched='live feed'):
    object_det = []
    if(alphabet.get() == 1):
        object_det.append('alphabet')
    if(sign.get() == 1):
        object_det.append('sign')
    if(searched!='live feed'):
        text = searched.get()
    if(val==2):
        live_feed(object_det)
    #give 'text' and array 'object_det' as input to code file
    #<enter function name here>#
        



app = ObjDet()
app.mainloop()

