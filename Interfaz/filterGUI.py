# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 19:32:58 2018

@author: SEBASTIAN CASTRO
"""

import sys
from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QApplication, QDialog, QMainWindow, QWidget, QFileDialog
from PyQt5.uic import loadUi
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg as FigureCanvas,
                                                NavigationToolbar2QT as NavigationToolbar)
import os
import numpy as np
import matplotlib.pyplot as plt
import pydicom 
from scipy import signal
from scipy import ndimage

## Filter functions
def medianF(img, k=3):
    x=np.ones(k)
    kernel,_ = np.meshgrid(x,x)
    kernel = kernel*(1/(k*k))
    imconv = np.copy(img)
    for i in range(1,len(img[:,1])-2):
        for j in range(1,len(img[1,:])-2):
            image_patch = img[i:i+len(kernel[1,:]),j:j+len(kernel[:,1])] 
            imconv[i][j]= (image_patch*kernel).sum()
    return imconv[i][j]

def gaussian(img,n=3,var=1):
    size =(n-1)/2
    #kernel = np.((n,n))
    x = np.linspace(-size,size,n)
    xv,yv = np.meshgrid(x,x)
    kernel = np.exp(-(xv**2 + yv**2)/(2*(var**2)))
    
    gaus =  signal.convolve2d(img,kernel, mode= 'same')
    return gaus

def laplace(img, A=1):
    A = ((10-A)/10)-1
    kernel = np.array([[0,-1,0],[-1,A+4,-1],[0,-1,0]])
    lapl =  signal.convolve2d(img,kernel, mode= 'same')
    return lapl

def isotropic(img,C= 0.1, t= 5):
    C = 1-((10-C)/10)
    lapla= ndimage.laplace(img)
    iso= img + (lapla*C)
    for i in range (0,t-1):
         iso = iso + (ndimage.laplace(iso)*C)
    
    return iso

def anisotropic(img,K= 1, t=10):
    K = 0.1-((10-K)/100)
    gx,gy = np.gradient( img )
    C = np.exp(-(abs(gx**2 + gy**2))/(K**2))
    anis = np.copy(img)
    for i in range (0,t):
        gx,gy = np.gradient( anis )
        gxgx,_ = np.gradient( C*gx )
        _,gygy = np.gradient( C*gy )
        anis = anis + gxgx + gygy
    
    return anis

def fourier (img):
    f_img = np.fft.fft2(img)
    s_img = np.fft.fftshift(f_img)
    return s_img

def ideal_filter(img, tipo = 'High Pass Filter', Fc=20):
    fft_img = fourier (img)
    abs_img = np.abs(fft_img)    
    a,b = np.where(abs_img == np.max(abs_img))
    size = np.shape(img)
    x = np.linspace(0,size[0],size[1])
    xv,yv = np.meshgrid(x,x)
    D = np.sqrt((xv-a)**2+(yv-b)**2)
    if tipo == 'High Pass Filter':
        D[D <= Fc] = 1
        D[D > Fc] = 0
        fft_fil = D*fft_img
    if  tipo == 'Low Pass Filter':
        D[D <= Fc] = 0
        D[D > Fc] = 1
        fft_fil = D*fft_img
        
    img_fil =np.fft.ifft2( np.fft.ifftshift(fft_fil) )
    img_fil = np.abs(img_fil)
    return img_fil

def butterworth(img, tipo = 'High Pass Filter', Fc=20, n = 2):
    fft_img = fourier (img)
    abs_img = np.abs(fft_img) 
    size = np.shape(img)
    a,b = np.where(abs_img == np.max(abs_img))
    x = np.linspace(0,size[0],size[1])
    xv,yv = np.meshgrid(x,x)
    D = np.sqrt((xv-a)**2+(yv-b)**2)
    if tipo == 'Low Pass Filter':
        H = 1 / (1 + (D/Fc)**(2*n))
        fft_fil = H*fft_img
    if  tipo == 'High Pass Filter':
        H = 1 / (1 + (Fc/D)**(2*n))
        fft_fil = H*fft_img
        
    img_fil =np.fft.ifft2( np.fft.ifftshift(fft_fil) ).real
    #img_fil = np.abs(img_fil)
    return img_fil

def frec_gaus(img, tipo = 'High Pass Filter', Fc=20):
    fft_img = fourier (img)
    abs_img = np.abs(fft_img) 
    size = np.shape(img)
    a,b = np.where(abs_img == np.max(abs_img))
    x = np.linspace(0,size[0],size[1])
    xv,yv = np.meshgrid(x,x)
    D = np.sqrt((xv-a)**2+(yv-b)**2)
    if tipo == 'Low Pass Filter':
        H = np.exp(-((D**2)/(2*Fc**2)))
        fft_fil = H*fft_img
    if  tipo == 'High Pass Filter':
        H = 1-np.exp(-((D**2)/(2*Fc**2)))
        fft_fil = H*fft_img
        
    img_fil =np.fft.ifft2( np.fft.ifftshift(fft_fil) ).real
    #img_fil = np.abs(img_fil)
    return img_fil    
#Parameter Window
##################################################################
class Parameters(QDialog):
    def __init__(self, parent=None):
        super(Parameters ,self).__init__(parent)
        loadUi('parameters.ui',self)
        self.filtrar.clicked.connect(self.Return)
        
        #Vertical slider 1
        self.verticalSlider1.setMinimum(0)
        self.verticalSlider1.setMaximum(10)
        self.verticalSlider1.setSingleStep(1)
        self.verticalSlider1.setValue(5)
        self.verticalSlider1.valueChanged.connect(self.getValueVertical1)
        
        #Vartical slider 2
        self.verticalSlider2.setMinimum(0)
        self.verticalSlider2.setMaximum(20)
        self.verticalSlider2.setSingleStep(1)
        self.verticalSlider2.setValue(10)
        self.verticalSlider2.valueChanged.connect(self.getValueVertical2)
        
        #Dial 1
        self.dial1.setMinimum(0)
        self.dial1.setMaximum(250)
        self.dial1.setSingleStep(10)
        self.dial1.setValue(0)
        self.dial1.valueChanged.connect(self.getValueDial)
        
    def getValueVertical1(self):
        global slider_value1
        slider_value1 = self.verticalSlider1.value()
        self.par1.setText(str(slider_value1))
        
    def getValueVertical2(self):
        global slider_value2
        slider_value2 = self.verticalSlider2.value()
        self.par2.setText(str(slider_value2))
        
    def getValueDial(self):
        global dial_value
        dial_value = self.dial1.value()
        self.cutoff.setText(str(dial_value))
        
    def Return(self):
        global FFT_type
        FFT_type = self.fftType.currentText()
        global order
        order = self.sp.value()
        self.close()

        

#Main Window
######################################################################
class MainWindows(QWidget):
    def __init__(self, parent=None):
        super(MainWindows,self).__init__(parent)
        loadUi('GUI_main.ui',self) 
        self.open.clicked.connect(self.openFileNameDialog)
        self.accept.clicked.connect(self.getItem)
        self.filtrar.clicked.connect(self.FilterFunction)
    
    def openFileNameDialog(self):    
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","All Files (*);;Python Files (*.py)", options=options)
        if fileName:
    #            print(fileName)
            self.imagen = pydicom.read_file(fileName)
            self.imagen = np.array(self.imagen.pixel_array)
            self.imagen[np.isnan(self.imagen)] = 0
            self.imagen = self.imagen - self.imagen.min()
            self.imagen = self.imagen / float(self.imagen.max())
            fig = Figure()
            ax1f = fig.add_subplot(111)
            ax1f.imshow( self.imagen, cmap='gray')
            ax1f.axis('off')
            ax1f.set_title('Original Image')
            try:
                self.clear1()
            except: 
                self.a=1
            self.addmpl(fig)  #funcion que une la figura a plotear con la interfaz
            

    
    def addmpl(self, fig):
        self.canvas = FigureCanvas(fig)
        self.plotorg.addWidget(self.canvas) #parte por defecto 
        self.canvas.draw()
       # self.toolbar = NavigationToolbar(self.canvas, self.org, coordinates=True) #para poner la barra de navegación sobre imagen zoom y otros
        #self.plotorg.addWidget(self.toolbar) #para añadirla
        
    def addmpl2(self, fig):
        self.canvas1 = FigureCanvas(fig)
        self.plotfilt.addWidget(self.canvas1) #parte por defecto 
        self.canvas1.draw()
        #self.toolbar = NavigationToolbar(self.canvas, self.proc, coordinates=True) #para poner la barra de navegación sobre imagen zoom y otros
        #self.plotproc.addWidget(self.toolbar) #para añadirla

    def clear1(self,):
        self.plotorg.removeWidget(self.canvas)
        self.canvas.close()
        
    def clear2(self,):
        self.plotfilt.removeWidget(self.canvas1)
        self.canvas1.close()
        #self.plotproc.removeWidget(self.toolbar1)
        #self.toolbar1.close()

        
    def getItem(self):
        global filter_type
        filter_type = self.FilterType.currentText()
        #self.close()
        self.Open = Parameters()
        self.Open.show()

    def FilterFunction(self):
        if filter_type == 'Median filter':
            self.image_filter=medianF(self.imagen,slider_value1)
        if filter_type == 'Gaussian filter':
            self.image_filter=gaussian(self.imagen,slider_value1,slider_value2)
        if filter_type == 'Laplace filter':
            self.image_filter=laplace(self.imagen,slider_value1)
        if filter_type == 'Isotropic filter':
            self.image_filter=isotropic(self.imagen,slider_value1,slider_value2)
        if filter_type == 'Anisotropic filter':
            self.image_filter=anisotropic(self.imagen,slider_value1,slider_value2)
        if filter_type == 'FFT ideal filter':
            self.image_filter=ideal_filter(self.imagen,FFT_type,dial_value)
        if filter_type == 'Butterword filter':
            self.image_filter=butterworth(self.imagen,FFT_type,dial_value,order)
        if filter_type == 'FFT gaussian filter':
            self.image_filter=frec_gaus(self.imagen,FFT_type,dial_value)
        
        fig1 = Figure()
        ax2f = fig1.add_subplot(111)
        ax2f.imshow( self.image_filter, cmap='gray')
        ax2f.axis('off')
        ax2f.set_title(filter_type)
        try:
             self.clear2()
        except: 
            self.a=1
        self.addmpl2(fig1)  #funcion que une la figura a plotear con la interfaz 
            
            
        
app = QApplication(sys.argv)
main = MainWindows()
main.show()
sys.exit(app.exec_())
        