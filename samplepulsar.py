from scintools.scint_sim import Simulation
import time
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from scipy.signal import convolve2d

start = time.time()
sim = Simulation(ds=0.1,ny=626,nx=126)
end = time.time()
print(end-start)

# Gives Fresnel radius
# Testing parameters are for J0030+0451, at distance = 329 kpc observed at f = 815.73 MHz (wavelength = 36.6 cm)
ds = 329*u.pc
k = 2*np.pi/(36.6*u.cm)
r_f = (np.sqrt((ds)/(k))).to(u.cm)
# Gives the pixel scale. dx overrides default ds value in the simulation
print((sim.dx*r_f*u.rad/(ds)).to(u.mas))

# Plots phase variation
plt.imshow(sim.xyp,aspect='auto',cmap='RdBu')
plt.xlabel(r'$x/r_F$')
plt.ylabel(r'$y/r_F$')
plt.title('Phase Variation')
plt.colorbar()

# Gets phase gradient (in angle/length)
# k = factor to convert into angle units (microarcsec)
phsgrad = (np.gradient(sim.xyp,r_f.value*sim.dx,r_f.value*sim.dy))*u.deg/(k.value)
phsgrad = phsgrad.to(u.uas)

#The actual shift should be averaged over the scattering disk size (~1 Fresnel radius)
radius = int(1/sim.dx) # 1 fresnel scale in pixels

#Makes circular gaussian for taking average
scat_disk = np.zeros((6*radius,6*radius))
x = np.linspace(-3*radius,3*radius,6*radius)
y = np.linspace(-3*radius,3*radius,6*radius)
X,Y = np.meshgrid(x,y)

scat_disk = np.exp(-((X)**2/(2*radius*radius) + (Y)**2/(2*radius*radius)))

im_shift_x = convolve2d(phsgrad[0],scat_disk,mode='same')/np.sum(scat_disk)
im_shift_y = convolve2d(phsgrad[1],scat_disk,mode='same')/np.sum(scat_disk)

# Two plots for the gradient in phase for both x and y along y=60
fig,ax = plt.subplots(2,layout='constrained')
#Gradient in phase in x-direction for path along y = 60
ax[0].plot(im_shift_x[60,:])
ax[0].set_xlabel(r'$x/r_F$')
ax[0].set_ylabel(r'Image Shift in x ($\mu$as)')
#Gradient in phase in y-direction for path along y = 60
ax[1].plot(im_shift_y[60,:])
ax[1].set_xlabel(r'$x/r_F$')
ax[1].set_ylabel(r'Image Shift in y ($\mu$as)')

plt.show()
