import pandas as pd
import numpy as np
import math, time, copy
import matplotlib.pyplot as plt

def Make_img(histo, Xbins, Xmin, Xmax, Ybins, Ymin, Ymax):
    img = np.zeros((100, 100), dtype=np.float32)
    
    for i in range(int(Ybins)):
        for j in range(int(Xbins)):
            img[i, j] = histo[i][j]#histo[i*(int(Xbins)+2)+j]
    #img = img[1:-1, 1:-1]
    return img
    
def Show2Dimg(img, title='CSC occupancy'):
    fig = plt.figure(figsize =(8, 8))
    img_temp = copy.deepcopy(img)
    cmap = plt.cm.jet
    cmap.set_under(color='white')
    max_=np.max(img_temp)
    img_temp[img_temp==0] = np.nan
    plt.imshow(img_temp, cmap=cmap, vmin=0.0000000001, vmax=max_)
    plt.colorbar()
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.show()
    del img_temp
    #plt.savefig('CSC_occupancy.png')
    
def Show1Dimg(vx, vy, xfit=None, yfit=None, x=r"Lumi [10$^{33}$ cm$^{-2}$ s$^{-1}$]", y="Occupancy (Hits/LS)", eymin=4, eymax=4, marker='.', line=False):
    plt.figure(figsize=(7, 4))
    if line==True:
        plt.axvline(x=9, color='red', linestyle='--')
    plt.plot(vx, vy, marker=marker, linestyle='', markersize=5, label='Data')
    if yfit is not None:
        plt.plot(xfit, yfit, color='red', label='Fit')
        plt.legend()
    plt.xlabel(x, size='14')
    plt.ylabel(y, size='14')
    #plt.ylim(0, 2000000)
    plt.grid(True)
    plt.ticklabel_format(axis="y", style="sci", scilimits=(eymin,eymax))
    plt.rc('xtick', labelsize='12')
    plt.rc('ytick', labelsize='12')
    plt.title(r'$\mathbf{CMS}\ \mathit{Private\ work}$', x=0.24, y=1.0, size=14)
    #plt.title('CMS', fontweight='bold',x=0.12, y=1.0, size=14)
    plt.title('2023 (13.6 TeV)',loc='right', size=14)
    plt.legend()
    plt.show()
    
def PLots_in_training(X, Xreco):
    print(' >> original image:')
    img = X[0][0].cpu().numpy()
    Show2Dimg(img)
    print(' >> AE-reco image:')
    img_reco = Xreco[0][0].detach().cpu().numpy()
    Show2Dimg(img_reco)
    print(' >> loss map:')
    img_loss = F.mse_loss(Xreco[0], X[0], reduction='none')[0].detach().cpu().numpy()
    Show2Dimg(img_loss)
