import numpy as np
import matplotlib.pyplot as plt

nmax = 1000 #Dimensions of the area. The area is assumed to be a cube
Ang = 1e-10 #An angstrom

side_r = 20*Ang #The length of an edge of the cube
dr = side_r/nmax #Length of each array position

def r_array(psi,nmax):
    '''
        Creates an array containing the r positions of each array position
    '''
    xi, yi, zi = np.indices(np.shape(psi),dtype = "float64")
    xi -= nmax//2*np.ones_like(psi,dtype = "float64")
    yi -= nmax//2*np.ones_like(psi,dtype = "float64")
    zi -= nmax//2*np.ones_like(psi,dtype = "float64")
    r_array = np.sqrt(xi*xi+yi*yi+zi*zi)*dr
    return r_array

def apply_boundary(psi,rmax,nmax):
    rarray = r_array(psi,nmax)
    small_enough = (rarray < rmax)
    psi = psi * small_enough
    return psi


psi = np.ones((nmax,nmax,nmax),dtype = "float64")
i = 0
hbar = 1.054e-34
m = 9.1093e-31
q = 1.6021e-19
Q = q
k = 8.8541e-12
E = 13.4*q

h2m = hbar**2/(2*m)
rmax = Ang*2
r_positions = r_array(psi,nmax)
zero_positions = r_positions == 0
no_zero = r_positions + zero_positions
r_positions += np.amin(no_zero)*1e-3*zero_positions


while True:
    psi_new = np.zeros_like(psi, dtype = "float64")
    psi_new[1:-1,1:-1,1:-1] = (psi[2:,1:-1,1:-1]+psi[:-2,1:-1,1:-1]+psi[1:-1,2:,1:-1]\
                              +psi[1:-1,:-2,1:-1]+psi[1:-1,1:-1,2:]+psi[1:-1,1:-1,:-2])\
                              *(-h2m)/(E-k*q*Q/r_positions[1:-1,1:-1,1:-1]-6*h2m)
    psi_new = apply_boundary(psi_new,rmax,nmax)
    i += 1
    if np.linalg.norm(psi-psi_new) < 0.1 or i > 1e3:
        break
    psi[:,:,:] = psi_new
psi = psi/np.sqrt(np.sum(np.abs(psi)))
psi2= psi*psi
plt.imshow(psi2[:,:,nmax//2],cmap="turbo")
