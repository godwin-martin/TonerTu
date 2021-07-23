import numpy as np
from numpy.fft import fft2, ifft2
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib import animation
import matplotlib.cm as cm
plt.rcParams['figure.figsize']=[12,12]
plt.rcParams.update({'font.size':18})


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


#Setting the discretization parameters
N = 10      #number of spatial gridpoints
L = 1      #domain length
tmax = 10   #time for which solution is obtained
dt = 0.1     #time step
dx = L/N     #gridsize

t = np.arange(0,tmax,dt)

#Defining the spatial and Fourier space meshes
x = y = np.linspace(0,L,N)
X, Y = np.meshgrid(x,y)

kx = ky = 2* np.pi* np.fft.fftfreq(N, d = dx)
KX, KY = np.meshgrid(kx,ky)


def time_deriv(field,t, v_0, alpha, beta, K, v1):
    D, px, py = np.split(field,3)
    D = np.reshape(D,(N,N))
    px = np.reshape(px,(N,N))
    py = np.reshape(py,(N,N))

    p_sqrd = px*px + py*py

    D_t = -v_0*(x_deriv(D*px) + y_deriv(D*py))

    px_t = -(alpha+ beta*p_sqrd)*px + K*Laplacian(px) -v1* x_deriv(D)
    py_t = -(alpha+ beta*p_sqrd)*py + K*Laplacian(py) -v1* y_deriv(D)

    D_lhs = np.ravel(D_t)
    px_lhs = np.ravel(px_t)
    py_lhs = np.ravel(py_t)

    return np.concatenate([D_lhs, px_lhs, py_lhs])



field0=np.ones((3,N,N))

field0[0] = np.exp(-5*((X-L/2)**2 + (Y-L/2)**2))
field0[1] = np.random.rand(N,N)
field0[2] = np.random.rand(N,N)


field0 = np.ravel(field0)


#fig, ax = plt.subplots()
#im = ax.imshow(field0[0], interpolation='bilinear', cmap=cm.RdYlGn,
               #origin='lower', extent=[-3, 3, -3, 3])



sol = odeint(time_deriv, field0, t, args = (1.1,-1,1,1,0.2))

re_sol = np.reshape(sol,(int(tmax/dt),3,N,N))

#Consistency check : We check if the total density remains constant in time
rhobar = []
for i in range(0,int(tmax/dt)):
    u = np.sum(re_sol[i,0])
    rhobar.append(u)

#np.shape(rhobar)

#plt.plot(t,rhobar)
std = np.std(rhobar)
ave = np.average(rhobar)
print('The error in the total density is =', std, '\n', 'The average total density is =', ave, 'The relative error is=', std/ave)

#to check if the density is taking negative signs
u = [np.min(re_sol[i,0]) for i in range(0,int(tmax/dt))]
v = [np.average(np.absolute(re_sol[i,0])) for i in range(0,int(tmax/dt))]
print(np.min(u),np.average(v))


#Animating

plot_args = {'cmap': 'viridis','vmin' : 0, 'vmax' : 2, 'linewidth': 0}



# Initialize line
fig = plt.figure(figsize=(10,8), dpi=200)
ax = fig.gca(projection='3d')
ax.set_zlim(-1, 5)
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$ \\rho$')
ax.set_title('Density with finite polarisation coupling')


DenPol = ax.plot_surface(X, Y, re_sol[0,0], **plot_args)



# Generate each animation frame
def animate(i):
    global DenPol
    
    u = re_sol[i,0]
        
    DenPol.remove()
    DenPol = ax.plot_surface(X, Y, u, **plot_args)
    return DenPol,

# Generate MatPlotLib FuncAnimation
disp = animation.FuncAnimation(fig, animate, frames=len(t), interval=1000*dt)
disp.save('Den_coup_pol.mp4')

U = re_sol[0,1] 
V = re_sol[0,2]

fig, ax = plt.subplots(1,1)
Q = ax.quiver(X, Y, U, V, pivot='mid', color='r')

ax.set_xlim(0, L)
ax.set_ylim(0, L)

def update_quiver(num, Q, X, Y):
    """updates the horizontal and vertical vector components by a
    fixed increment on each frame
    """

    U = re_sol[num,1]
    V = re_sol[num,2]

    Q.set_UVC(U,V)

    return Q,

# you need to set blit=False, or the first set of arrows never gets
# cleared on subsequent frames
anim = animation.FuncAnimation(fig, update_quiver, fargs=(Q, X, Y), frames = len(t),
                               interval=1000*dt, blit=False)
anim.save('quiver.mp4')
fig.tight_layout()
plt.show()




