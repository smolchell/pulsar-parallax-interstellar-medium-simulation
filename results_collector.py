import numpy as np
import os
from pulsar_obs_fit import *
from scipy.optimize import curve_fit
import pandas as pd
import csv
import time
start_time = time.time()

### read in csv file as panda data frame
df = pd.read_csv('shortlisted_pulsar_data.csv')

# convert name, ra and dec col to strings
df['Jname'] = df['Jname'].astype('string')
df['ra'] = df['ra'].astype('string')
df['dec'] = df['dec'].astype('string')

headers = df.columns.to_list()
additionalHeaders = ["start", "stop", "tru_fit_ang", "tru_fit_pmra","tru_fit_pmdec", "tru_err_ang", "tru_err_pmra","tru_err_pmdec", "obs_ang", "obs_pmra","obs_pmdec","obs_err_ang", "obs_err_pmra","obs_err_pmdec"]
headers = headers + additionalHeaders

# formula for pulsar fit
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

times = [1,1.5,2,2.5,3] ## reduce upper limit to 3 years
with open('scintil_applied_results_period_variation.csv', 'w', encoding='UTF8', newline='') as resultFile:
    writer = csv.writer(resultFile)
    # write the header
    writer.writerow(headers)

    for sim_num in range(10):
        print(f"---- simulation {sim_num} started ----")
        x_file_name = "C://Users//mitch//OneDrive - Swinburne University//Ashley Stock's files - simulations//im_shift_x_sim_"+str(sim_num)+".npy"
        y_file_name = "C://Users//mitch//OneDrive - Swinburne University//Ashley Stock's files - simulations//im_shift_y_sim_"+str(sim_num)+".npy"
        x_data = np.load(x_file_name)
        y_data = np.load(y_file_name)
        for pulsar_num in range(len(df)):
            pulsar = df.loc[pulsar_num]

            ra = pulsar['ra']
            dec = pulsar['dec']
            pmra = pulsar['Pmra']
            pmdec = pulsar['Pmdec']
            dist = pulsar['dist']
            wavlgth = (299792458/pulsar['f'])*100.0

            start = 0
            for stop in times:
                obvsPeriod, dt = observation_Bounds(start,stop)

                #Will need to read in dist and wavlgth from your text file for this
                pixel_scale, k , r_f = give_pixel_scale(dist, wavlgth, dx=0.1)
                pxd, pm_ra_cosdec, pm_dec,c = pulsar_coord(ra, dec, pmra, pmdec, obvsPeriod,dist)
                dra, ddec = get_pulsar_position(obvsPeriod, pxd, pm_ra_cosdec, pm_dec,c,dt)
                psr_x_pix, psr_y_pix = fit_sim(dra,ddec,pixel_scale) #output restricts the what we're looking at in the simulation

                popt, pcov = curve_fit(fit_pulsar_astrometry,np.concatenate((dt.value,dt.value)),np.concatenate((dra.to(u.mas).value,ddec.to(u.mas).value)))

                tru_fit_ang, tru_fit_pmra, tru_fit_pmdec = popt
                tru_err_ang, tru_err_pmra, tru_err_pmdec = np.sqrt(np.diag(pcov))
                # Below prints the best fit for parallax angle and proper motions of ra and dec, as well as their respective errors
                # print("True: Best fit parallax angle  ra proper motion, dec proper motion", popt)
                # print("True: Error in parallax angle  ra proper motion, dec proper motion", np.sqrt(np.diag(pcov)))

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

                obs_fit_ang, obs_fit_pmra, obs_fit_pmdec = obs_popt
                obs_err_ang, obs_err_pmra, obs_err_pmdec = np.sqrt(np.diag(obs_pcov))
                #should save the obs_popt and errors to a file
                # print("Observed: Best fit parallax angle, ra proper motion, dec proper motion", obs_popt, "mas")
                # print("Observed: Error in parallax angle, ra proper motion, dec proper motion", np.sqrt(np.diag(obs_pcov)), "mas")

                results = [tru_fit_ang, tru_fit_pmra, tru_fit_pmdec,
                        tru_err_ang, tru_err_pmra, tru_err_pmdec,
                        obs_fit_ang, obs_fit_pmra, obs_fit_pmdec,
                        obs_err_ang, obs_err_pmra, obs_err_pmdec]

                new_row = pulsar.to_list()+ [start,stop] + results
                writer.writerow(new_row)
                resultFile.flush()
        print(f"---- simulation {sim_num} complete ----")
        print("--- time elapsed %.2f seconds ---" % (time.time() - start_time))

print("--- total run time %.2f seconds ---" % (time.time() - start_time))
