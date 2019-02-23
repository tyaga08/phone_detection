import numpy as np
import cv2
import sklearn as sk

class find_hog_of_image:
    def __init__(self, win_size, block_x, block_y, cell_size):
        self.winSize = win_size
        self.blockSize = (block_y, block_x) #Keep an eye
        self.blockStride = (block_y//2, block_x//2) #Keep an eye
        self.cellSize = cell_size #Keep an eye
        self.nbins = 9
        self.derivAperture = 1
        self.winSigma = -1.
        self.histogramNormType = 0
        self.L2HysThreshold = 0.2
        self.gammaCorrection = 1
        self.nlevels = 64
        self.signedGradients = False #Keep an eye
    
    def compute_hog_descriptor(self,img):
        self.hog = cv2.HOGDescriptor(self.winSize,self.blockSize,self.blockStride,self.cellSize,self.nbins,self.derivAperture,self.winSigma,self.histogramNormType,self.L2HysThreshold,self.gammaCorrection,self.nlevels, self.signedGradients)
        self.descriptor = self.hog.compute(img)

