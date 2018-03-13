import numpy as np
import astropy
from astropy.io import fits, ascii
from astropy.table import Table, Column
import matplotlib.pyplot as plt
from scipy import optimize, stats
import pandas as pd
import re

from astropy.coordinates import ICRS, Distance, Angle, SkyCoord
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=71, Om0=0.27)

def correct_rgc(coord, glx_ctr = SkyCoord(0.0, 0.0, unit='arcsec'), 
			glx_PA=Angle('37d42m54s'), glx_incl=Angle('77.5d')):
	# distance from coord to glx centre
	sky_radius = glx_ctr.separation(coord)
	avg_dec = 0.5 * (glx_ctr.dec + coord.dec).radian
	x = (glx_ctr.ra - coord.ra) * np.cos(avg_dec)
	y = glx_ctr.dec - coord.dec
	# azimuthal angle from coord to glx  -- not completely happy with this
	phi = glx_PA - Angle('90d') + Angle(np.arctan(y.arcsec / x.arcsec), unit=u.rad)
	# convert to coordinates in rotated frame, where y-axis is galaxy major ax; 
	# have to convert to arcmin b/c can't do sqrt(x^2+y^2) when x and y are angles
	xp = (sky_radius * np.cos(phi.radian)).arcsec
	yp = (sky_radius * np.sin(phi.radian)).arcsec
	# de-project
	ypp = yp / np.cos(glx_incl.radian)
	obj_radius = np.sqrt(xp ** 2 + ypp ** 2)  # in arcsec

	return obj_radius

Table_drp = Table.read('/home/qliu/MaNGA/drpall-v2_1_2.fits')
table_drp = Table(Table_drp.columns["plateifu","nsa_sersic_n","nsa_sersic_ba"]).to_pandas()
table_drp['NUV_r'] = Table_drp.columns['nsa_sersic_absmag'][:,1] - Table_drp.columns['nsa_sersic_absmag'][:,4]

# Build pure SFG sample
table_pip3D = Table.read('/home/qliu/MaNGA/pipe3d/manga.Pipe3D-v2_1_2.fits').to_pandas()

print "Merging catalogs... MaNGA DRP"
table_merg = pd.merge(table_pip3D,table_drp)
table_face = table_merg[table_merg.nsa_sersic_ba>0.5]    # face-on (b/a > 0.5)

# Classification from D4000
table_D4000 = ascii.read('/home/qliu/MaNGA/beta_profile_D4000.dat').to_pandas()
data_sample = pd.merge(table_face,table_D4000)

# Use Flag from image (Remove merger, center-shifted)
table_use = ascii.read('/home/qliu/MaNGA/beta_use_image.dat').to_pandas()
data_sample = pd.merge(table_use[table_use.flag==1], data_sample)
 

# ----Analysis-----#
table_all = data_sample[data_sample.D4000_mode!=0]
table_io = data_sample[data_sample.D4000_mode==1]
table_oi = data_sample[data_sample.D4000_mode==-1]

# PSF
table_all=table_all[table_all.re_arc>2.5]
table_io=table_io[table_io.re_arc>2.5]
table_oi=table_oi[table_oi.re_arc>2.5]

# Setting
Redshift = pd.Series(np.array(table_all.redshift)) # redshifts
Mode_SFG = pd.Series(np.array(table_all.D4000_mode)) # D4000 modes
Distance_lum = pd.Series(np.array(table_all.dl))  # Luminosity Distance in Mpc
Re_kpc = pd.Series(np.array(table_all.re_kpc))  # Re in kpc
Gal_PA = pd.Series(np.array(table_all.pa))  	# Patch Angle
Gal_ba = pd.Series(np.array(table_all.nsa_sersic_ba))	# Axis Ratio b/a

dir=["/home/qliu/MaNGA/SFG_pipe3d/%s.Pipe3D.cube.fits.gz"%ID for ID in table_all.mangaid]


# Start From Here
Plateifu = np.array([])
D4000_Mode = np.array([])
Vel = np.array([])
V_dis = np.array([])

num = 6
for i,f in enumerate(dir[:num]):
	ID = re.findall(r'\d+?-\d*e?\d*?',f)[0]

	print "Processing manga%s... %d galaxies left"%(ID,len(dir)-i)

	try:
		hdu = fits.open(f)
	except IOError:
		print "manga%s datacube not included...skip"%ID
		continue

	# Read SSP maps
	ssp = hdu[1].data  
	Binning_map = ssp[1]    
	Vel_map = ssp[13];	V_dis_map = ssp[15]

	shape = Binning_map.shape[0]
	bins = np.unique(Binning_map)

	# Read Absorption maps
	absp = hdu[4].data 
	D4000_map = absp[5]

	# spaxel scale in kpc
	Mpc_to_cm = 1e8*astropy.constants.pc.value   #1 Mpc in cm
	dl_kpc = Distance_lum[i] * 1e3	# distance in kpc from Mpc
	dl_cm = Distance_lum[i] * Mpc_to_cm   # distance in cm
	angle_spaxel = (0.5/3600)*(np.pi/180)  # The angle of spaxel is 0.5''
	spaxel_kpc = angle_spaxel * dl_kpc   # spaxel scale in kpc
	re_kpc = Re_kpc[i]   # re in kpc
	re_spaxel = re_kpc/spaxel_kpc # re in spaxel

	# Remove Small FOv
	fov_re = shape/2./re_spaxel
	if fov_re < 2.:  # clip FOV < 2 Re
		print "manga%s fov < 2 Re... skip"%ID; continue

	yy,xx = np.meshgrid(np.arange(0,shape,1),np.arange(0,shape,1))
	center = (shape-1)/2.
	dist_map = np.sqrt((yy-center)**2+(xx-center)**2) # Distance in pixel

	dist_map_arcs = dist_map * 0.5 # Distance in arcsecond
	zp = 10000.  #zero point of RA,DEC center
	glx_ctr = SkyCoord(zp, zp, unit='arcsec') # center position
	glx_pa = Angle(Gal_PA[i]*u.deg)  # pitch angle
	glx_incl = Angle(np.arccos(Gal_ba[i])*u.rad)  #inclination arccos(b/a)

	dist_cor_map = np.zeros_like(dist_map)  
	for ix in range(shape):
		x_loc,y_loc = (xx[ix]-center), (yy[ix]-center)
		coord = SkyCoord(0.5*x_loc + zp, 0.5*y_loc + zp, unit='arcsec')
		r_cor_arc = correct_rgc(coord, glx_ctr=glx_ctr,glx_PA=glx_pa,glx_incl=glx_incl)    
		dist_cor_map[ix] = r_cor_arc/0.5

	# Reset batch quantities for usable bins
	r_val = np.array([])
	vel_val = np.array([])
	v_dis_val = np.array([])

	for j, b in enumerate(bins[1:]):
		ms_all = Ms_map[Binning_map==b]
		use_bin = (ms_all!=0) & (ms_all==ms_all)
		if sum(use_bin)==0: j-=1; continue

		r_spaxel = np.median(dist_cor_map[Binning_map==b][use_bin]) # radial distance in spaxel
		r_re = r_spaxel/re_spaxel 	# radial distance in re

		d4000 = np.median(D4000_map[Binning_map==b][use_bin])
		vel = np.median(Vel_map[Binning_map==b][use_bin])
		v_dis = np.median(V_dis_map[Binning_map==b][use_bin])
		
		r_val = np.append(r_val,r_re)
		vel_val = np.append(vel_val,vel)
		v_dis_val = np.append(v_dis_val,v_dis)

	
	v_rec = cosmo.H0 * cosmo.comoving_distance(Redshift[i])
	v_val = vel_val - v_rec.value
	np.sum(v_val)

	# Add to lists
	Vel = np.append(Vel, v_val)
	V_dis = np.append(V_dis, v_dis_val)
	Plateifu = np.append(Plateifu, [ID for d in range(len(r_val))])
	D4000_Mode = np.append(D4000_Mode, [Mode_SFG[i] for d in range(len(r_val))])

	hdu.close()
