from scintools.scint_sim import Simulation
import time
import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, get_body_barycentric
import astropy.units as u
from scipy.signal import convolve2d
from scipy.optimize import curve_fit

## Simulated scintillation shift ##
start = time.time()
sim = Simulation(ds=0.1,ny=626,nx=126)
end = time.time()
print(end-start)


# Testing parameters are for J0030+0451, at distance = 329 kpc observed at f = 815.73 MHz (wavelength = 36.6 cm)
dist = 329 #Mhz
wavlgth = 36.6 #cm

def phase_calculations(sim, dist, wavlgth):
    # Gives Fresnel radius
    ds = (dist/2)*u.pc # distance to screen (distance to pulsar / 2, unless specified otherwise)
    k = 2*np.pi/(wavlgth*u.cm)
    r_f = (np.sqrt((ds)/(k))).to(u.cm)

    # Gives the pixel scale. dx overrides default ds value in the simulation
    pixel_scale = (sim.dx*r_f*u.rad/(ds)).to(u.mas)
    print("One pixel = ", pixel_scale)

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
    return scat_disk, phsgrad, pixel_scale

scat_disk, phsgrad, pixel_scale = phase_calculations(sim, dist, wavlgth)

## original position of convol

## Find true sky position with fitted curve ##
#Define your observing times
dt = np.linspace(0,1/100,100)*u.yr
t = Time('2026-10-03 05:11:23.5') + dt

#Define the coordinates of your pulsar
c = SkyCoord(ra='00:30:27.4', dec='+04:51:39.7', unit=(u.hourangle,u.deg),
             frame='icrs',obstime=t[0])

pxd=(3.040*u.mas).to(u.pc,u.parallax()) #parallax distance
pm_ra_cosdec=-6.150*u.mas/u.yr #proper motion in right ascension
pm_dec=0.370*u.mas/u.yr #proper motion in declination

# Function to get dra and ddec
def get_pulsar_position(t,pxd,pm_ra_cosdec,pm_dec):
    earthpos = get_body_barycentric('earth',t)
    dra = (earthpos.x*np.sin(c.ra) - earthpos.y*np.cos(c.ra))/(pxd*np.cos(c.dec))*u.rad #parallax signal
    dra += pm_ra_cosdec*dt/np.cos(c.icrs.dec) #add the proper motion
    ddec = ((earthpos.x*np.cos(c.ra) + earthpos.y*np.sin(c.ra))*np.sin(c.dec) - earthpos.z*np.cos(c.dec))*u.rad/pxd  #parallax signal
    ddec += pm_dec*dt #add the proper motion

    return dra, ddec

dra, ddec = get_pulsar_position(t,pxd,pm_ra_cosdec,pm_dec)

# Function to fit curve, advised that units be hard-coded to avoid errors
def fit_pulsar_astrometry(dt,pxd,pmra,pmdec):
    t = Time('2026-10-03 05:11:23.5') + dt[:dt.shape[0]//2]*u.yr
    d=(pxd*u.mas).to(u.pc,u.parallax())
    pm_ra_cosdec = pmra*u.mas/u.yr
    pm_dec = pmdec*u.mas/u.yr

    earthpos = get_body_barycentric('earth',t)

    dra = (earthpos.x*np.sin(c.ra) - earthpos.y*np.cos(c.ra))/(d*np.cos(c.dec))*u.rad #parallax signal
    dra += pm_ra_cosdec*dt[:dt.shape[0]//2]*u.yr/np.cos(c.icrs.dec) #add the proper motion
    dra = dra.to(u.mas).value

    ddec = ((earthpos.x*np.cos(c.ra) + earthpos.y*np.sin(c.ra))*np.sin(c.dec) - earthpos.z*np.cos(c.dec))*u.rad/d  #parallax signal
    ddec += pm_dec*dt[:dt.shape[0]//2]*u.yr #add the proper motion
    ddec = ddec.to(u.mas).value

    return np.concatenate((dra,ddec))

popt, pcov = curve_fit(fit_pulsar_astrometry,np.concatenate((dt.value,dt.value)),np.concatenate((dra.to(u.mas).value,ddec.to(u.mas).value)))

# Below prints the best fit for parallax angle and proper motions of ra and dec, as well as their respective errors
print("True: Best fit parallax angle  ra proper motion, dec proper motion", popt)
print("True: in parallax angle  ra proper motion, dec proper motion", np.sqrt(np.diag(pcov)))

## Fit observed pulsar position ##
# Conversions to fit the true positions to simulation units and apply offsets to remove negative values
psr_x_pix = np.round(dra.to(u.mas)/pixel_scale).astype(int)
psr_x_pix -= min(psr_x_pix)
psr_x_pix = np.clip (psr_x_pix, 0, 125) #applies bounds to remain in the simulation size

psr_y_pix = np.round(ddec.to(u.mas)/pixel_scale).astype(int)
psr_y_pix -= min(psr_y_pix)
psr_y_pix = np.clip (psr_y_pix, 0, 625) #applies bounds to remain in the simulation size

### split code here as convolution takes most time
im_shift_x = convolve2d(phsgrad[0],scat_disk,mode='same')/np.sum(scat_disk)
im_shift_y = convolve2d(phsgrad[1],scat_disk,mode='same')/np.sum(scat_disk)

# Apply the shifts at the pulsar positions
psr_shifts_x = im_shift_x[psr_x_pix, psr_y_pix]
psr_shifts_y = im_shift_y[psr_x_pix, psr_y_pix]

# Get the observed positions by taking true positions and adding the corresponding shifts
obs_pos_x = (dra).to(u.mas) + (psr_shifts_x)*u.uas
obs_pos_y = (ddec).to(u.mas) + (psr_shifts_y)*u.uas

obs_popt, obs_pcov = curve_fit(fit_pulsar_astrometry,np.concatenate((dt.value,dt.value)),np.concatenate(((obs_pos_x).to(u.mas).value,(obs_pos_y).to(u.mas).value)))

print("Observed: Best fit parallax angle, ra proper motion, dec proper motion", obs_popt, "mas")
print("Observed: Error in parallax angle, ra proper motion, dec proper motion", np.sqrt(np.diag(obs_pcov)), "mas")

# Plots true position along with observed position of the pulsar
plt.scatter((dra).to(u.mas),(ddec).to(u.mas),s=1,label='True Position')
plt.scatter((obs_pos_x).to(u.mas),(obs_pos_y).to(u.mas),s=1,label='Observed Position')
plt.xlabel('Relative Right Ascension (mas)')
plt.ylabel('Relative Declination (mas)')
plt.xlim(2,-15) #right ascension increases to the left
plt.legend()

plt.show()
