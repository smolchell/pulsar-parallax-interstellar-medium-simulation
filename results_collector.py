import numpy as np
import os
from pulsar_obs_fit import *
from scipy.optimize import curve_fit

# path = "C://Users//mitch//OneDrive - Swinburne University//Ashley Stock's files - simulations"
# dir_list = os.listdir(path)
# dir_list = [f for f in dir_list if os.path.isfile(path+'/'+f)]
# filenames = []
# for x in dir_list:
#     if x.endswith(".npy"):
#         # Prints only text file present in My Folder
#         print(path+"//"+x)
#         filenames.append(path+"//"+x)

# data = []
# for file in filenames:
#     data.append(np.load(file))

x_data = np.load("C://Users//mitch//OneDrive - Swinburne University//Ashley Stock's files - simulations//im_shift_x_sim_0.npy")
y_data = np.load("C://Users//mitch//OneDrive - Swinburne University//Ashley Stock's files - simulations//im_shift_y_sim_0.npy")
# print(np.shape(x_data))

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

ra = '00:30:27.4'
dec ='+04:51:39.7'
pmra = -6.150
pmdec = 0.370

start = 0
stop = 3

dist = 329 #Mhz
wavlgth = 36.6 #cm

obvsPeriod, dt = observation_Bounds(start,stop)

#Will need to read in dist and wavlgth from your text file for this
pixel_scale, k , r_f = give_pixel_scale(dist, wavlgth, dx=0.1)
pxd, pm_ra_cosdec, pm_dec,c = pulsar_coord(ra, dec, pmra, pmdec, obvsPeriod)
dra, ddec = get_pulsar_position(obvsPeriod, pxd, pm_ra_cosdec, pm_dec,c,dt)
psr_x_pix, psr_y_pix = fit_sim(dra,ddec,pixel_scale) #output restricts the what we're looking at in the simulation

popt, pcov = curve_fit(fit_pulsar_astrometry,np.concatenate((dt.value,dt.value)),np.concatenate((dra.to(u.mas).value,ddec.to(u.mas).value)))

# Below prints the best fit for parallax angle and proper motions of ra and dec, as well as their respective errors
print("True: Best fit parallax angle  ra proper motion, dec proper motion", popt)
print("True: in parallax angle  ra proper motion, dec proper motion", np.sqrt(np.diag(pcov)))

#----------Loop over simulations--------------------------------------------

#will need to change the names of these files so it loops over the simulations
im_shift_x = ((x_data*u.deg)/(k.value*r_f.value)).to(u.uas)
im_shift_y = ((y_data*u.deg)/(k.value*r_f.value)).to(u.uas)

# Apply the shifts at the pulsar positions
psr_shifts_x = im_shift_x[psr_x_pix, psr_y_pix]
psr_shifts_y = im_shift_y[psr_x_pix, psr_y_pix]

# Get the observed positions by taking true positions and adding the corresponding shifts
# obs_pos_x = (dra).to(u.mas) + ((psr_shifts_x)*u.uas)
# obs_pos_y = (ddec).to(u.mas) + ((psr_shifts_y)*u.uas)
obs_pos_x = (dra).to(u.mas) + psr_shifts_x
obs_pos_y = (ddec).to(u.mas) + psr_shifts_y

obs_popt, obs_pcov = curve_fit(fit_pulsar_astrometry,np.concatenate((dt.value,dt.value)),np.concatenate(((obs_pos_x).to(u.mas).value,(obs_pos_y).to(u.mas).value)))

#should save the obs_popt and errors to a file
print("Observed: Best fit parallax angle, ra proper motion, dec proper motion", obs_popt, "mas")
print("Observed: Error in parallax angle, ra proper motion, dec proper motion", np.sqrt(np.diag(obs_pcov)), "mas")
