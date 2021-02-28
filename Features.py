from sklearn.neighbors import KNeighborsClassifier
from cv2 import *
import numpy as np
from math import *
from matplotlib import pyplot as plt
import csv
import skimage.measure    
from skimage.restoration import estimate_sigma
import glob
import time as t
start = t.time()
def estimate_noise(img):
    	return estimate_sigma(img, multichannel=True, average_sigmas=True)
"""
def imag(img,string):
	b,g,r = cv2.split(img)
	im = cv2.merge([r,g,b])
	x,y,z = np.shape(im)
	red = np.zeros((x,y,z),dtype=int)
	green = np.zeros((x,y,z),dtype=int)
	blue = np.zeros((x,y,z),dtype=int)
	for i in range(0,x):
	    for j in range(0,y):
	        red[i][j][0] = im[i][j][0]
	        green[i][j][1]= im[i][j][1]
	        blue[i][j][2] = im[i][j][2]
	with open('RED.csv', mode='a') as Red:
    		mean, std = meanStdDev(red)
    		Red = csv.writer(Red, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    		Red.writerow([str(img[..., 0].min()), str(img[..., 0].max()),str(mean[0][0]),str(std[0][0]),str(np.sum(red<=255)),estimate_noise(img),skimage.measure.shannon_entropy(red),int(string)])
	with open('GREEN.csv', mode='a') as Green:
    		mean, std = meanStdDev(green)
    		Green = csv.writer(Green, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    		Green.writerow([str(img[..., 1].min()), str(img[..., 1].max()),str(mean[1][0]),str(std[1][0]),str(np.sum(red<=255)),estimate_noise(img),skimage.measure.shannon_entropy(blue),int(string)])
	with open('BLUE.csv', mode='a') as Blue:
    		mean, std = meanStdDev(blue)
    		Blue = csv.writer(Blue, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    		Blue.writerow([str(img[..., 2].min()), str(img[..., 2].max()),str(mean[2][0]),str(std[2][0]),str(np.sum(red<=255)),estimate_noise(img),skimage.measure.shannon_entropy(green),int(string)])
	k = waitKey(0)
"""
def imag(img,string):
	mean = np.mean(img)
	SD = np.std(img)
	with open('Features.csv', mode='a') as features:
    		features = csv.writer(features, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    		features.writerow([str(img[...,].min()), str(img[...,].max()),str(mean),str(SD), str(np.sum(img<=255)), estimate_noise(img), skimage.measure.shannon_entropy(img),int(string) ])
	k = waitKey(0)
with open('Features.csv', mode='w') as features:
    	features = csv.writer(features, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    	features.writerow(['Minmum Pixel Intensity', 'Maximum Pixel Intensity', 'Mean','SD','Total Pixel','Noise','Entropy','Result'])
"""
with open('RED.csv', mode='w') as features:
    	features = csv.writer(features, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    	features.writerow(['Minmum Pixel Intensity', 'Maximum Pixel Intensity', 'Mean','SD','Total Pixel','Noise','Entropy','Result'])
with open('GREEN.csv', mode='w') as features:
    	features = csv.writer(features, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    	features.writerow(['Minmum Pixel Intensity', 'Maximum Pixel Intensity', 'Mean','SD','Total Pixel','Noise','Entropy','Result'])
with open('BLUE.csv', mode='w') as features:
    	features = csv.writer(features, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    	features.writerow(['Minmum Pixel Intensity', 'Maximum Pixel Intensity', 'Mean','SD','Total Pixel','Noise','Entropy','Result'])
"""
for img in glob.glob("/home/lenovo/Downloads/indoor/gt/*.png"):
    n= cv2.imread(img)
    imag(n,"0")
for img in glob.glob("/home/lenovo/Downloads/indoor/hazy/*.png"):
    n= cv2.imread(img)
    imag(n,"1")
end = t.time()
print(end-start)
