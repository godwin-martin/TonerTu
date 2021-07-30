import numpy as np
import time
start = time.process_time()
from numpy.fft import fft2, ifft2
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib import animation
import matplotlib.cm as cm
plt.rcParams['figure.figsize']=[12,12]
plt.rcParams.update({'font.size':18})


#Setting the discretization parameters
N = 10     #number of spatial gridpoints
L = 2      #domain length
tmax = 10   #time for which solution is obtained
dt = 0.1    #time step
dx = L/N    #gridsize

#Defining the spatial and Fourier space meshes
x = y = np.linspace(0,L,N)
X, Y = np.meshgrid(x,y)

kx = ky = 2* np.pi* np.fft.fftfreq(N, d = dx)
KX, KY = np.meshgrid(kx,ky)

#Function to compute the x derivative
def x_deriv(Field):
    Fieldhat = fft2(Field)
    Field_xhat = (1j)*KX*Fieldhat
    Field_x = np.real(ifft2(Field_xhat))
    return Field_x

#Function to compute the y derivative
def y_deriv(Field):
    Fieldhat = fft2(Field)
    Field_yhat = (1j)*KY*Fieldhat
    Field_y = np.real(ifft2(Field_yhat))
    return Field_y

#Function to compute the scalar laplacian
def Laplacian(Field):
    Fieldhat = fft2(Field)
    Lap_Fieldhat = - (KX**2 + KY**2) * Fieldhat
    Lap_Field = np.real(ifft2(Lap_Fieldhat))
    return Lap_Field

#Function to compute the $\partial_t$ term at every instance of time
def time_deriv(field,t,Dprime,Lambda,kappa1,kappa2,Z):
    D, px, py = np.split(field,3)

    D = np.reshape(D,(N,N)); px = np.reshape(px,(N,N)); py = np.reshape(py,(N,N))

    p_squared = px*px + py*py

    #derivatives
    px_x = x_deriv(px); px_y = y_deriv(px)
    py_x = x_deriv(py); py_y = y_deriv(py)

    Dpx_x = x_deriv(D*px); Dpy_y = y_deriv(D*py)

    lap_px = Laplacian(px); lap_py = Laplacian(py)

    D_x = x_deriv(D); D_y = y_deriv(D)

    p_sqrd_x = x_deriv(p_squared); p_sqrd_y = y_deriv(p_squared)

    div_p = px_x + py_y; div_Dp = Dpx_x + Dpy_y

    D_RHS = -div_Dp + Dprime*Laplacian(D)
    px_RHS = -Lambda*(px*px_x + py*px_y)-((1-D)+(p_squared))*px + kappa1*lap_px +kappa2*x_deriv(div_p)-Z*D_x 
    py_RHS = -Lambda*(px*py_x + py*py_y)-((1-D)+(p_squared))*py + kappa2*lap_py +kappa2*y_deriv(div_p)-Z*D_y
    

    D_t = np.ravel(D_RHS)
    px_t = np.ravel(px_RHS)
    py_t = np.ravel(py_RHS)

    return np.concatenate([D_t, px_t, py_t])

t = np.arange(0,tmax,dt)

#Initial conditions

field0=np.ones((3,N,N))

field0[0] = 4*np.exp(-0.5*((X-L/2)**2 + (Y-L/2)**2))
field0[1] = np.random.rand(N,N)
field0[2] = np.random.rand(N,N)


field0 = np.ravel(field0)

Dprime = 0.4; Lambda = 0.5; kappa2 = 0.5; 

start =0
end = 1
step = 0.5

SOL = []
for Z in np.arange(start,end,step):
    for kappa1 in np.arange(start,end,step):
        sol = odeint(time_deriv, field0, t, args = (Dprime, Lambda, kappa1, kappa2, Z))
        re_sol = np.reshape(sol,(int(tmax/dt),3,N,N))
        SOL.append(re_sol)

np.save(f'Dprime={Dprime},Lambda={Lambda},k1={kappa1},k2={kappa2},Z={Z}.npy',SOL)

print(time.process_time() - start)