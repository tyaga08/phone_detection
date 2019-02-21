import numpy as np
import cv2
import sklearn as sk

class find_hog:
    def __init__(self, win_size, block_size):
        self.winSize = (win_size,win_size)
        self.blockSize = (block_size,block_size) #Keep an eye
        self.blockStride = (block_size//2,block_size//2) #Keep an eye
        self.cellSize = (10,10) #Keep an eye
        self.nbins = 9
        self.derivAperture = 1
        self.winSigma = -1.
        self.histogramNormType = 0
        self.L2HysThreshold = 0.2
        self.gammaCorrection = 1
        self.nlevels = 64
        self.signedGradients = False #Keep an eye
    
    def compute_hog_descriptor(self,img):
        hog = cv2.HOGDescriptor(self.winSize,self.blockSize,self.blockStride,self.cellSize,self.nbins,self.derivAperture,self.winSigma,self.histogramNormType,self.L2HysThreshold,self.gammaCorrection,self.nlevels, self.signedGradients)
        descriptor = hog.compute(img)

