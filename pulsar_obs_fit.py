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

#--------------Functions for supercomputer--------------#

def create_sintil_screen(ds, ny,nx, debug = False):
    start = time.time()
    sim = Simulation(ds=ds,ny=ny,nx=nx)
    end = time.time()
    if debug: print(end-start)
    return sim

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

def shift_x_y(phsgrad,scat_disk):
    im_shift_x = convolve2d(phsgrad[0],scat_disk,mode='same')/np.sum(scat_disk)
    im_shift_y = convolve2d(phsgrad[1],scat_disk,mode='same')/np.sum(scat_disk)
    return im_shift_x, im_shift_y

#--------------------------Pulsar-dependent values----------------------------------------------

def give_pixel_scale(dist, wavlgth,dx=0.1):
    # Gives Fresnel radius
    ds = (dist/2)*u.pc # distance to screen (distance to pulsar / 2, unless specified otherwise)
    k = 2*np.pi/(wavlgth*u.cm)
    r_f = (np.sqrt((ds)/(k))).to(u.cm)

    pixel_scale = (dx*r_f*u.rad/(ds)).to(u.mas)
    return pixel_scale, k, r_f

# Function to get dra and ddec
def get_pulsar_position(obvsPeriod,pxd,pm_ra_cosdec,pm_dec,c,dt):
    earthpos = get_body_barycentric('earth',obvsPeriod)
    dra = (earthpos.x*np.sin(c.ra) - earthpos.y*np.cos(c.ra))/(pxd*np.cos(c.dec))*u.rad #parallax signal
    dra += pm_ra_cosdec*dt/np.cos(c.icrs.dec) #add the proper motion
    ddec = ((earthpos.x*np.cos(c.ra) + earthpos.y*np.sin(c.ra))*np.sin(c.dec) - earthpos.z*np.cos(c.dec))*u.rad/pxd  #parallax signal
    ddec += pm_dec*dt #add the proper motion

    return dra, ddec

## Fit observed pulsar position ##
# assuming a pixel is 0.1 frensel scale then the pixel scale can be calculated -> there will be slight variation and should be calculated independently for each simulation
def fit_sim(dra,ddec,pixel_scale=0.1,bounds=[12000,20000]): #if no pixel scale given then 0.1 is the default
    # Conversions to fit the true positions to simulation units and apply offsets to remove negative values
    psr_x_pix = np.round(dra.to(u.mas)/pixel_scale).astype(int)
    psr_x_pix -= min(psr_x_pix)
    psr_x_pix = np.clip (psr_x_pix, 0, bounds[0]) #applies bounds to remain in the simulation size

    psr_y_pix = np.round(ddec.to(u.mas)/pixel_scale).astype(int)
    psr_y_pix -= min(psr_y_pix)
    psr_y_pix = np.clip (psr_y_pix, 0, bounds[1]) #applies bounds to remain in the simulation size
    return psr_x_pix, psr_y_pix

# Function to fit curve, advised that units be hard-coded to avoid errors
"""
def fit_pulsar_astrometry(dt,pxd,pmra,pmdec,c):
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
"""

## Find true sky position with fitted curve ##
#Define your observing times
###---SHIFT OBSERVATION BOUNDS HERE---###
def observation_Bounds(start, stop):
    dt = np.linspace(start,stop,365*stop)*u.yr
    obvsPeriod = Time('2026-10-03 05:11:23.5') + dt
    return obvsPeriod, dt


#-----------Loop over pulsars-----------------------------------------------

#---------------Read in file of pulsar properties for these-----------------
#Define the coordinates of your pulsar
def pulsar_coord(ra,dec, pmra, pmdec, obvsPeriod, dist):
    c = SkyCoord(ra, dec, unit=(u.hourangle,u.deg),
                frame='icrs',obstime=obvsPeriod[0])

    pxd=(dist*u.pc).to(u.pc,u.parallax()) #parallax distance #(1/dist*rads)*u.mas
    # pxd= dist*u.mas
    print(pxd)
    pm_ra_cosdec=pmra*u.mas/u.yr #proper motion in right ascension
    pm_dec=pmdec*u.mas/u.yr #proper motion in declination
    return pxd, pm_ra_cosdec, pm_dec,c

#------------------------------------------------------------#

###---test values---###
# scilination sim test
# test values
"""
#This would all be run on the supercomputer
ds=0.1
ny=626
nx=126
sim = create_sintil_screen(ds,ny,nx)
# Testing parameters are for J0030+0451, at distance = 329 kpc observed at f = 815.73 MHz (wavelength = 36.6 cm)
dist = 329 #Mhz
wavlgth = 36.6 #cm

scat_disk, phsgrad, pixel_scale = phase_calculations(sim, dist, wavlgth)
"""

"""
ra = '00:30:27.4'
dec='+04:51:39.7'
pmra = -6.150
pmdec = 0.370

start = 0
stop = 2

dist = 329 #Mhz
wavlgth = 36.6 #cm

obvsPeriod, dt = observation_Bounds(start,stop)

#Will need to read in dist and wavlgth from your text file for this
pixel_scale, k , r_f = give_pixel_scale(dist, wavlgth, dx=0.1)
pxd, pm_ra_cosdec, pm_dec,c = pulsar_coord(ra, dec, pmra, pmdec, obvsPeriod)
dra, ddec = get_pulsar_position(obvsPeriod, pxd, pm_ra_cosdec, pm_dec)
psr_x_pix, psr_y_pix = fit_sim(dra,ddec,pixel_scale) #output restricts the what we're looking at in the simulation

popt, pcov = curve_fit(fit_pulsar_astrometry,np.concatenate((dt.value,dt.value)),np.concatenate((dra.to(u.mas).value,ddec.to(u.mas).value)))

# Below prints the best fit for parallax angle and proper motions of ra and dec, as well as their respective errors
print("True: Best fit parallax angle  ra proper motion, dec proper motion", popt)
print("True: in parallax angle  ra proper motion, dec proper motion", np.sqrt(np.diag(pcov)))

#----------Loop over simulations--------------------------------------------

#will need to change the names of these files so it loops over the simulations
im_shift_x = (np.load('simulation_x.npy')*u.deg/(k.value*r_f.value)).to(u.uas)
im_shift_y = (np.load('simulation_y.npy')*u.deg/(k.value*r_f.value)).to(u.uas)

# Apply the shifts at the pulsar positions
psr_shifts_x = im_shift_x[psr_x_pix, psr_y_pix]
psr_shifts_y = im_shift_y[psr_x_pix, psr_y_pix]

# Get the observed positions by taking true positions and adding the corresponding shifts
obs_pos_x = (dra).to(u.mas) + (psr_shifts_x)*u.uas
obs_pos_y = (ddec).to(u.mas) + (psr_shifts_y)*u.uas

obs_popt, obs_pcov = curve_fit(fit_pulsar_astrometry,np.concatenate((dt.value,dt.value)),np.concatenate(((obs_pos_x).to(u.mas).value,(obs_pos_y).to(u.mas).value)))

#should save the obs_popt and errors to a file
print("Observed: Best fit parallax angle, ra proper motion, dec proper motion", obs_popt, "mas")
print("Observed: Error in parallax angle, ra proper motion, dec proper motion", np.sqrt(np.diag(obs_pcov)), "mas")


#---won't need this except for checking things look right----
# Plots true position along with observed position of the pulsar
plt.scatter((dra).to(u.mas),(ddec).to(u.mas),s=1,label='True Position')
plt.scatter((obs_pos_x).to(u.mas),(obs_pos_y).to(u.mas),s=1,label='Observed Position')
plt.xlabel('Relative Right Ascension (mas)')
plt.ylabel('Relative Declination (mas)')
plt.xlim(2,-15) #right ascension increases to the left
plt.legend()

plt.show()
"""