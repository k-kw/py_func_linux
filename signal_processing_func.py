import numpy as np
import matplotlib.pyplot as plt

#return absolute
def img_hpf(img,threshold):
    H=np.zeros(img.shape)
    for h in range(img.shape[0]):
        for w in range(img.shape[1]):
            if (((img.shape[0]/2)-h)**2+((img.shape[1]/2)-w)**2)**0.5>threshold :
                H[h,w]=1
    img=np.fft.fft2(img)
    img=np.fft.fftshift(img)
    img=img*H
    img=np.fft.fftshift(img)
    img=np.fft.ifft2(img)
    img=np.abs(img)
    return img