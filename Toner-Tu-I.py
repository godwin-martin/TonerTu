import numpy as np
from numpy.fft import fft2, ifft2
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib import animation
import matplotlib.cm as cm
plt.rcParams['figure.figsize']=[12,12]
plt.rcParams.update({'font.size':18})

#Setting the discretization parameters
N = 10   #number of spatial gridpoints
L = 1     #domain length
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
def time_deriv(field,t,A, B, C, E, F):
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

    px_RHS = -A*(px*px_x + py * px_y) - (B*(1-D)+ C*(p_squared))*px + lap_px - E*D_x + (F/2)* p_sqrd_x - F*px*(div_p)
    py_RHS = -A*(px*py_x + py * py_y) - (B*(1-D)+ C*(p_squared))*py + lap_py - E*D_y + (F/2)* p_sqrd_y - F*py*(div_p)

    D_t = np.ravel(div_Dp)
    px_t = np.ravel(px_RHS)
    py_t = np.ravel(py_RHS)

    return np.concatenate([D_t, px_t, py_t])

#Function to animate the density variation given the parameters A, B, C, E, F and the initial conditions

def density_anim(a, b, c, e, f, field0):
    A = a
    B = b
    C = c
    E = e
    F = f

    dt = np.abs(t[0] - t[1])
    sol = odeint(time_deriv, field0, t, args = (a,b,c,e,f))
    re_sol = np.reshape(sol,(int(tmax/dt),3,N,N))

    #Animating

    plot_args = {'cmap': 'viridis','vmin' : -20, 'vmax' : 20, 'linewidth': 0}



    # Initialize line
    fig = plt.figure(figsize=(10,8), dpi=200)
    ax = fig.gca(projection='3d')
    ax.set_zlim(0, 10)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$ \\rho$')
    ax.set_title('Density')


    TonerTu = ax.plot_surface(X, Y, re_sol[0,0], **plot_args)



    # Generate each animation frame
    def animate(i):
        nonlocal TonerTu
    
        u = re_sol[i,0]
        
        TonerTu.remove()
        TonerTu = ax.plot_surface(X, Y, u, **plot_args)
        return TonerTu,

    # Generate MatPlotLib FuncAnimation
    disp = animation.FuncAnimation(fig, animate, frames=len(t), interval=1000*dt)
    disp.save(f'Toner-Tu1_{a}_{b}_{c}_{e}_{f}.mp4')


# %%
#Function to animate the polarisation
def polar_anim(a,b,c,e,f,field0):
    dt = np.abs(t[0] - t[1])
    sol = odeint(time_deriv, field0, t, args = (a,b,c,e,f))
    re_sol = np.reshape(sol,(int(tmax/dt),3,N,N))

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
    anim.save(f'quiver_{a}_{b}_{c}_{e}_{f}.mp4')
    fig.tight_layout()
    plt.show()

t = np.arange(0,tmax,dt)

#Initial conditions

field0 = np.ones((3,N,N))
field0 = np.ravel(field0)


density_anim(1,0.1,0.1,0,0,field0)

polar_anim(1,0.1,0.1,0,0,field0)