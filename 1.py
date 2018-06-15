from scipy.ndimage.filters import convolve as filter2
from matplotlib.pyplot import figure,draw,pause,gca,show,axis,close,title
from mathutils.geometry import intersect_point_line
from mathutils import Vector
from termios import tcflush, TCIOFLUSH
import numpy as np
import cv2
import sys




HSKERN =np.array([[1/12, 1/6, 1/12],
                  [1/6,    0, 1/6], 
                  [1/12, 1/6, 1/12]],float)

kernelX = np.array([[-1, 1],
                     [-1, 1]]) * .25 #kernel for computing d/dx

kernelY = np.array([[-1,-1],
                     [ 1, 1]]) * .25 #kernel for computing d/dy

HSKERN2 =np.array([[1/12, 1/6, 1/12],
                  [1/6,    0, 1/6], 
                  [1/12, 1/6, 1/12]],float) * 2

kernelX2 = np.array([[-1, 1],
                     [-1, 1]]) * .4 #kernel for computing d/dx

kernelY2 = np.array([[-1,-1],
                     [ 1, 1]]) * .4 #kernel for computing d/dy

kernelT = np.ones((2,2))*.25
kernelT2 = np.ones((2,2))*.5

FRAME_BEGIN = 14

def calculate_UV(U,V,fx,fy,ft):
    for i in range(0,U.shape[0]):
        for j in range(0,U.shape[1]):   
            if fx[i][j] != 0 and fy[i][j] != 0 and  ft[i][j] != 0:
                line = ((-1 * ft[i][j] / fx[i][j], 0), (0, -1 * ft[i][j] / fy[i][j]))
                point = (0,0)
                a,__ = intersect_point_line(point, line[0], line[1])
                U[i][j] = a[0]
                V[i][j] = a[1]
    return U,V

def plotderiv(fx,fy,ft,msg):
    fg = figure(figsize=(18,5))
    title(msg)
    ax = fg.subplots(1,3)
    
    for f,a,t in zip((fx,fy,ft),ax,('$f_x$','$f_y$','$f_t$')):
        axis('off')
        h=a.imshow(f,cmap='bwr')
        a.set_title(t)
        fg.colorbar(h,ax=a)



def getimgfiles():
    imgs = []
    for i in range(FRAME_BEGIN, 52):
        path = './Data/img' + str(i) + '.png'
        img = cv2.imread(path,0)
        imgs.append(img)
    return imgs

def compareGraphs(u,v,Inew,title, scale:int=.4, quivstep:int=6):
    ax = figure().gca()
    ax.set_title(title)
    axis('off')
    ax.imshow(Inew,cmap = 'gray')
    shape = u.shape
    for i in range(0,shape[0], quivstep):
        for j in range(0,shape[1], quivstep):
            if abs(v[i,j] - u[i,j]) > 3:
                ax.arrow(j,i, v[i,j]*scale, u[i,j]*scale, color='green',head_width=4, head_length=6)

    draw();pause(0.01)
    

def horn_schunck():
    imgs= getimgfiles()
    
    for i in range(len(imgs)-1):
        if i > 0:
            tcflush(sys.stdin, TCIOFLUSH)
            input('Pressione qualquer tecla para avançar para o proximo quadro:\n')
            close('all')
        im1 = imgs[i]
        im2 = imgs[i+1]
        
        U,V,msg = HornSchunck(im1, im2, 1., 200, True, 1)
        compareGraphs(U,V, im2, msg)
        U,V,msg = HornSchunck(im1, im2, 1., 200, True, 2)
        compareGraphs(U,V, im2, msg)
        U,V,msg = HornSchunck(im1, im2, 1., 200, True, 3)
        compareGraphs(U,V, im2, msg)
        U,V,msg = HornSchunck(im1, im2, 1., 200, True, 4)
        compareGraphs(U,V, im2, msg)

    return U,V

def HornSchunck(im1, im2, alpha:float=0.001, Niter:int=8, verbose:bool=False, type:int=1):

    # im1: image at t=0
    # im2: image at t=1
    # alpha: regularization constant
    # Niter: number of iteration

    im1 = im1.astype(np.float32)
    im2 = im2.astype(np.float32)

    U = np.zeros([im1.shape[0],im1.shape[1]])
    V = np.zeros([im1.shape[0],im1.shape[1]])

    mensage= " Inicializated "
    if type == 1 or type == 2:
        [fx, fy, ft] = computeDerivatives(im1, im2)
        mensage += 'with Derivative 1 '
    if type == 3 or type == 4:
        [fx, fy, ft] = computeDerivatives2(im1, im2) 
        mensage += 'with Derivative 2 '
    if type == 2 or type == 4:
        U,V = calculate_UV(U,V,fx,fy,ft)
        mensage += 'and u,v equal to q (nearest point of the line) '
    else:
        mensage += 'and u,v equal to 0'

    if verbose:
        plotderiv(fx,fy,ft, mensage)
        print("[*]Tipo:" +  mensage +'\n  ')
        print('   Mean x ' + str(np.sum(cv2.mean(fx))) )
        print('   Mean y ' + str(np.sum(cv2.mean(fy))) )
        print('   Mean t ' + str(np.sum(cv2.mean(ft))) )

    for _ in range(Niter):
        uAvg = filter2(U, HSKERN)
        vAvg = filter2(V, HSKERN)
        der = (fx*uAvg + fy*vAvg + ft) / (alpha**2 + fx**2 + fy**2)
        U = uAvg - fx * der
        V = vAvg - fy * der

    return U,V,mensage


def computeDerivatives(im1, im2):
    global I
    fx = filter2(im1,kernelX) + filter2(im2,kernelX)
    fy = filter2(im1,kernelY) + filter2(im2,kernelY)

    ft = filter2(im1,kernelT) + filter2(im2,-kernelT)
    return fx,fy,ft

def computeDerivatives2(im1, im2):
    global I
    fx = filter2(im1,kernelX2) + filter2(im2,kernelX2)
    fy = filter2(im1,kernelY2) + filter2(im2,kernelY2)

    ft = filter2(im1,kernelT2) + filter2(im2,-kernelT2)
    return fx,fy,ft

if __name__ == '__main__':
    U,V = horn_schunck()
    show()