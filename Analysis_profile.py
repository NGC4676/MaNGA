import numpy as np
import astropy
from astropy.io import fits, ascii
from astropy.table import Table, Column
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import optimize, stats
import pandas as pd
import seaborn as sns
import re

from pylab import *

from astropy.coordinates import ICRS, Distance, Angle, SkyCoord
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=71, Om0=0.27)

rplot = np.linspace(0, 2.4, 10)

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

NII_plot=np.linspace(-1.5,0.0,200)
def BPT_border(log_NII_Ha,linetype = "Kauffmann"):
	if linetype=="Kauffmann": return 1.3 + 0.61 / (log_NII_Ha-0.05)
	elif linetype=="Kewley": return 1.19 + 0.61 / (log_NII_Ha - 0.47)

def BPT_distance(bpt_x,bpt_y,NII=NII_plot):
	OIII = BPT_border(NII)
	dists = np.sqrt((bpt_x-NII)**2+(bpt_y-OIII)**2)
	dist = np.min(dists)
	if (bpt_y < BPT_border(bpt_x)) & (bpt_x<NII_plot.max()): return dist
	else: return -1*dist

def k(lamda):
    if lamda > 0.63:
        k = 2.659*(-1.857+1.040/lamda)+4.05
    else:
        k = 2.659*(-2.156+1.509/lamda-0.198/lamda**2+0.011/lamda**3)+4.05
    return k

#--------Sample Selection--------#
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

Ms = np.array([])
SFR = np.array([])
Ha = np.array([])
Hb = np.array([])
OIII = np.array([])
NII = np.array([])
Radius = np.array([])
D4000 = np.array([])
EW_Ha = np.array([])
Vel = np.array([])
V_dis = np.array([])
Av = np.array([])

num = 600
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
	Ms_map = ssp[19];		Av_map = ssp[11]
	Vel_map = ssp[13];	V_dis_map = ssp[15]

	shape = Binning_map.shape[0]
	bins = np.unique(Binning_map)

	# Read emission maps
	flux_el = hdu[3].data
	Ha_map = flux_el[45];	Hb_map = flux_el[28]
	OIII_map = flux_el[26];	NII_map = flux_el[46]
	EWHa_map = flux_el[216]

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
	ms_val = np.array([])		# Sigma_Ms
	sfr_val = np.array([])		# Sigma_SFR

	ha_val = np.array([])
	hb_val = np.array([])
	oiii_val = np.array([])
	nii_val = np.array([])

	D4000_val = np.array([])
	ew_ha_val = np.array([])

	r_val = np.array([])
	vel_val = np.array([])
	v_dis_val = np.array([])
	Av_val = np.array([])

	for j, b in enumerate(bins[1:]):
		ms_all = Ms_map[Binning_map==b]
		use_bin = (ms_all!=0) & (ms_all==ms_all)
		if sum(use_bin)==0: j-=1; continue

		r_spaxel = np.median(dist_cor_map[Binning_map==b][use_bin]) # radial distance in spaxel
		r_re = r_spaxel/re_spaxel 	# radial distance in re

		ms = np.median(Ms_map[Binning_map==b][use_bin])
		ha = np.median(Ha_map[Binning_map==b][use_bin])
		hb = np.median(Hb_map[Binning_map==b][use_bin])
		oiii = np.median(OIII_map[Binning_map==b][use_bin])
		nii = np.median(NII_map[Binning_map==b][use_bin])

		tau = np.log((ha/hb)/2.86)
		Ebv = 1.086/(k(0.4861)-k(0.6563))*tau
		if Ebv<0: Ebv = 0 

		ha_corr = ha*10**(0.4*k(0.6563)*Ebv)
		hb_corr = hb*10**(0.4*k(0.4861)*Ebv)
		oiii_corr = oiii*10**(0.4*k(0.5007)*Ebv)
		nii_corr = nii*10**(0.4*k(0.6583)*Ebv)

		L_ha = ha_corr * (4*np.pi*dl_cm**2) * 1e-16   # flux to Lum
		lg_sfr = np.log10(L_ha) - 41.10	# Kennicutt 1998

		d4000 = np.median(D4000_map[Binning_map==b][use_bin])
		ew_ha = np.median(EWHa_map[Binning_map==b][use_bin])
		vel = np.median(Vel_map[Binning_map==b][use_bin])
		v_dis = np.median(V_dis_map[Binning_map==b][use_bin])
		av = np.median(Av_map[Binning_map==b][use_bin])

		ms_val = np.append(ms_val,ms)
		sfr_val = np.append(sfr_val,lg_sfr)

		ha_val = np.append(ha_val,ha_corr)
		hb_val = np.append(hb_val,hb_corr)
		oiii_val = np.append(oiii_val,oiii_corr)
		nii_val = np.append(nii_val,nii_corr)
		
		r_val = np.append(r_val,r_re)
		D4000_val = np.append(D4000_val,d4000)
		ew_ha_val = np.append(ew_ha_val,ew_ha)
		vel_val = np.append(vel_val,vel)
		v_dis_val = np.append(v_dis_val,v_dis)
		Av_val = np.append(Av_val,av)

	# Add to lists
	Ms = np.append(Ms, ms_val)
	SFR = np.append(SFR, sfr_val)
	Ha = np.append(Ha, ha_val)
	Hb = np.append(Hb, hb_val)
	OIII = np.append(OIII, oiii_val)
	NII = np.append(NII, nii_val)
	Radius = np.append(Radius, r_val)
	EW_Ha = np.append(EW_Ha, ew_ha_val)
	D4000 = np.append(D4000, D4000_val)
	v_rec = cosmo.H0 * cosmo.comoving_distance(Redshift[i])
	Vel = np.append(Vel, vel_val-v_rec.value)
	V_dis = np.append(V_dis, v_dis_val)
	Av = np.append(Av, Av_val)
	Plateifu = np.append(Plateifu, [ID for d in range(len(r_val))])
	D4000_Mode = np.append(D4000_Mode, [Mode_SFG[i] for d in range(len(r_val))])

	hdu.close()

# Save Data
data_profile = Table([Plateifu, D4000_Mode, Radius, Ms, SFR, Ha, Hb, OIII, NII, D4000, EW_Ha, Vel, V_dis, Av], names=['plateifu','D4000_mode','r','Sigma_Ms', 'Sigma_SFR', 'ha', 'hb', 'oiii', 'nii', 'D4000','EW_Ha','v','sigma','Av'])
ascii.write(data_profile, 'beta_profile.dat',overwrite=True)
#data_profile = data_profile[data_profile.EW_Ha>data_profile.EW_Ha.quantile(0.001)]





## Start From Here ##

data_profile = ascii.read('beta_profile.dat').to_pandas()
data_profile['Sigma_SSFR'] = data_profile.Sigma_SFR - data_profile.Sigma_Ms
data_profile['log_EW_Ha'] = log10(-data_profile.EW_Ha)

data_profile['v_s'] = abs(data_profile.v/data_profile.sigma)
#data_profile['bpt_x'] = log10(data_profile.nii/data_profile.ha)
#data_profile['bpt_y'] = log10(data_profile.oiii/data_profile.hb)
#data_profile['bpt_dist'] = [BPT_distance(bpt_x,bpt_y,NII=NII_plot) for (bpt_x,bpt_y) in zip(data_profile.bpt_x,data_profile.bpt_y)]
data_profile['Sigma_gas'] = np.log10(15*data_profile.Av + (8.743+0.462*data_profile.nii -12 -2.67))
data_profile['gas_frac'] = 10**data_profile.Sigma_gas/(10**data_profile.Sigma_gas+10*data_profile.Sigma_Ms)
data_profile['t_dep'] = data_profile.Sigma_gas - data_profile.Sigma_SFR - 9

profile_io = data_profile[data_profile.D4000_mode==1]
profile_oi = data_profile[data_profile.D4000_mode==-1]


# Plotting
def median_fitting(x,y,d=10):
	y = y[(x>0)&(x<2.5)]
	x = x[(x>0)&(x<2.5)]
	xq = np.linspace(x.min(),x.max(),d+1)
	xp = [(xq[k]+xq[k+1])/2. for k in range(d)]
	yp = np.array([y[(x>xq[k])&(x<xq[k+1])].median() for k in range(d)]).ravel()
	yp_a = np.array([y[(x>xq[k])&(x<xq[k+1])].quantile(0.3) for k in range(d)]).ravel()
	yp_b = np.array([y[(x>xq[k])&(x<xq[k+1])].quantile(0.7) for k in range(d)]).ravel()
	#cof = np.polyfit(xp,yp,1)
	return xp,yp,yp_a,yp_b

t = 'log_EW_Ha'
figure(figsize=(8,6))
ax = plt.subplot(111)
rm1,tm1,tm1a,tm1b = median_fitting(profile_io.r, profile_io[t],d=10)
rm2,tm2,tm2a,tm2b = median_fitting(profile_oi.r, profile_oi[t],d=10)
plt.plot(rm1, tm1,c='r', ls='-',lw=5,alpha=1.,zorder=2)
plt.plot(rm2, tm2,c='deepskyblue',ls='-',lw=5,alpha=1.,zorder=2)
fill_between(x=rm1,y1=tm1a,y2=tm1b,color='r',alpha=0.5,label='Inside-out')
fill_between(x=rm2,y1=tm2a,y2=tm2b,color='skyblue',alpha=0.5,label='Outside-in')
legend(loc='best',fontsize=15)
xlabel(r'R $\rm [R_e]$',fontsize=15)
ylabel(r'log $\rm EW(H{\alpha}) [\AA]$',fontsize='large')
xlim(0.15,2.); ylim(0.6,2.4); 
tight_layout()


# Control Sample
    # M
Ctrl_io = pd.DataFrame()
Ctrl_oi = pd.DataFrame()
hist_io,_ = np.histogram(table_fit_io.log_mass,range=(8.75,11.5),bins=11)
hist_oi,_ = np.histogram(table_fit_oi.log_mass,range=(8.75,11.5),bins=11)
mp = np.linspace(8.875,11.375,11)
for m in mp:
    bin_io=table_fit_io[abs(table_fit_io.log_mass-m)<0.125]
    bin_oi=table_fit_oi[abs(table_fit_oi.log_mass-m)<0.125]
    ctrl_num=min(len(bin_io),len(bin_oi))
    ctrl_io=bin_io.sample(ctrl_num)
    ctrl_oi=bin_oi.sample(ctrl_num)
    Ctrl_io = pd.concat([Ctrl_io,ctrl_io])
    Ctrl_oi = pd.concat([Ctrl_oi,ctrl_oi])

C_io=pd.merge(profile_io,Ctrl_io,on="plateifu")
C_oi=pd.merge(profile_oi,Ctrl_oi,on="plateifu")
    
t = 'log_EW_Ha'
figure(figsize=(10,5))
gs = mpl.gridspec.GridSpec(1, 2, width_ratios=[3,2])

ax1 = plt.subplot(gs[0])
rm1,tm1,tm1a,tm1b = median_fitting(C_io.r, C_io[t],d=10)
rm2,tm2,tm2a,tm2b = median_fitting(C_oi.r, C_oi[t],d=10)
plt.plot(rm1, tm1,c='r', ls='-',lw=5,alpha=1.,zorder=2,label="IO M-Control")
plt.plot(rm2, tm2,c='deepskyblue',ls='-',lw=5,alpha=1.,zorder=2,label='OI M-Control')
fill_between(x=rm1,y1=tm1a,y2=tm1b,color='r',alpha=0.5)
fill_between(x=rm2,y1=tm2a,y2=tm2b,color='skyblue',alpha=0.5)
xlabel(r'R $\rm [R_e]$',fontsize=15)
ylabel(r'log $\rm EW(H{\alpha}) [\AA]$',fontsize=15)
xlim(0.15,2.); ylim(0.6,2.4); 
rm1,tm1,tm1a,tm1b = median_fitting(profile_io.r, profile_io[t],d=10)
rm2,tm2,tm2a,tm2b = median_fitting(profile_oi.r, profile_oi[t],d=10)
plt.plot(rm1, tm1,c='r', ls='--',lw=3,alpha=.8,zorder=2,label='IO All')
plt.plot(rm2, tm2,c='deepskyblue',ls='--',lw=3,alpha=.8,zorder=2,label='OI All')
legend(loc='best',fontsize="large")

ax2 =plt.subplot(gs[1])
sns.distplot(Ctrl_oi.log_mass,color="r",label='IO M-Control')
sns.distplot(Ctrl_io.log_mass,color="steelblue",label='OI M-Control')
legend(loc='best',fontsize='large')
xlabel(r'$\rm{log{\ }M_*[M_{\odot}]}$',fontsize=15)
tight_layout()
#plt.savefig("../Pub/EWHa_profile_Mcontrol.pdf",dpi=400)

    # Morph
Ctrl_io = pd.DataFrame()
Ctrl_oi = pd.DataFrame()
hist_io,_ = np.histogram(table_fit_io.lambda_re,range=(0.1,1.),bins=9)
hist_oi,_ = np.histogram(table_fit_oi.lambda_re,range=(0.1,1.),bins=9)
lamp = np.linspace(0.15,0.95,9)
for lam in lamp:
    bin_io=table_fit_io[abs(table_fit_io.lambda_re-lam)<0.05]
    bin_oi=table_fit_oi[abs(table_fit_oi.lambda_re-lam)<0.05]
    ctrl_num=min(len(bin_io),len(bin_oi))
    ctrl_io=bin_io.sample(ctrl_num)
    ctrl_oi=bin_oi.sample(ctrl_num)
    Ctrl_io = pd.concat([Ctrl_io,ctrl_io])
    Ctrl_oi = pd.concat([Ctrl_oi,ctrl_oi])

sns.distplot(Ctrl_oi.lambda_re)
sns.distplot(Ctrl_io.lambda_re)
    
C_io=pd.merge(profile_io,Ctrl_io,on="plateifu")
C_oi=pd.merge(profile_oi,Ctrl_oi,on="plateifu")

figure(figsize=(10,5))
gs = mpl.gridspec.GridSpec(1, 2, width_ratios=[3,2])

ax1 = plt.subplot(gs[0])
rm1,tm1,tm1a,tm1b = median_fitting(C_io.r, C_io[t],d=10)
rm2,tm2,tm2a,tm2b = median_fitting(C_oi.r, C_oi[t],d=10)
plt.plot(rm1, tm1,c='r', ls='-',lw=5,alpha=1.,zorder=2,label='IO $\lambda$-Control')
plt.plot(rm2, tm2,c='deepskyblue',ls='-',lw=5,alpha=1.,zorder=2,label='OI $\lambda$-Control')
fill_between(x=rm1,y1=tm1a,y2=tm1b,color='r',alpha=0.5)
fill_between(x=rm2,y1=tm2a,y2=tm2b,color='skyblue',alpha=0.5)
legend(loc='best',fontsize='large')
xlabel(r'R $\rm [R_e]$',fontsize=15)
ylabel(r'log $\rm EW(H{\alpha}) [\AA]$',fontsize=15)
xlim(0.15,2.); ylim(0.6,2.4); 
rm1,tm1,tm1a,tm1b = median_fitting(profile_io.r, profile_io[t],d=10)
rm2,tm2,tm2a,tm2b = median_fitting(profile_oi.r, profile_oi[t],d=10)
plt.plot(rm1, tm1,c='r', ls='--',lw=3,alpha=.8,zorder=2,label='IO All')
plt.plot(rm2, tm2,c='deepskyblue',ls='--',lw=3,alpha=.8,zorder=2,label='OI All')
legend(loc='best',fontsize='large')

ax2 =plt.subplot(gs[1])
sns.distplot(Ctrl_oi.lambda_re,color="r",label='IO $\lambda$-Control')
sns.distplot(Ctrl_io.lambda_re,color="steelblue",label='OI $\lambda$-Control')
legend(loc='best',fontsize='large')
xlabel(r'$\rm \lambda_R(1.5{\ }Re)$',fontsize=15)
tight_layout()
plt.savefig("../Pub/EWHa_profile_Lcontrol.pdf",dpi=400)


# D4000
t = ['D4000']
figure(figsize=(7,6))
ax = plt.subplot(111)
rm1,tm1,tm1a,tm1b = median_fitting(profile_io.r, profile_io[t],d=10)
rm2,tm2,tm2a,tm2b = median_fitting(profile_oi.r, profile_oi[t],d=10)
plt.plot(rm1, tm1,c='r', ls='-',lw=5,alpha=1.,zorder=2)
plt.plot(rm2, tm2,c='deepskyblue',ls='-',lw=5,alpha=1.,zorder=2)
fill_between(x=rm1,y1=tm1a,y2=tm1b,color='r',alpha=0.5)
fill_between(x=rm2,y1=tm2a,y2=tm2b,color='skyblue',alpha=0.5)
xlabel('R (Re)'); ylabel('D4000')
xlim(0.1,2.); ylim(1.,1.8)
tight_layout()


#### depletion time ratio ####
group_cen = data_profile[data_profile.r<0.3].groupby(data_profile.plateifu)
group_cen.t_dep.median()

group_disk = data_profile[(data_profile.r>0.3)&(data_profile.r<1.2)].groupby(data_profile.plateifu)
group_disk.t_dep.median()
dep_ratio = group_cen.t_dep.median()-group_disk.t_dep.median()
temp = pd.DataFrame({'plateifu':dep_ratio.index,'dep_ratio':dep_ratio,'D4000_mode':group_cen.D4000_mode.median()})
tem1=temp[temp.D4000_mode==1]
tem2=temp[temp.D4000_mode==-1]
figure(figsize=(6,5))
sns.distplot(tem1.dropna().dep_ratio,label='Inside-out',color='firebrick')
sns.distplot(tem2.dropna().dep_ratio,label='Outside-in')
legend(loc='best',fontsize='large')
xlim(-2.5,1.5); ylim(0,2.)
xlabel(r'$\rm log(\tau_{cen}/\tau_{disk})$',fontsize='large')




####Integrated Metallicity#####
lines = pd.DataFrame({'plateifu':data_profile.plateifu,'D4000_mode':data_profile.D4000_mode,
				'ha':data_profile.ha,'hb':data_profile.hb,
				'oiii':data_profile.oiii,'nii':data_profile.nii})
lines=lines.dropna()
gp = lines.groupby("plateifu")
ha_tot = gp.ha.sum()
hb_tot = gp.hb.sum()
oiii_tot = gp.oiii.sum()
nii_tot = gp.nii.sum()


D4000_mode = gp.D4000_mode.mean()
O3N2 = np.log10(oiii_tot/hb_tot*ha_tot/nii_tot)
N2 = np.log10(nii_tot/ha_tot)
sns.distplot(8.533-0.214*O3N2[D4000_mode==1].dropna(),color='firebrick')
sns.distplot(8.533-0.214*O3N2[D4000_mode==-1].dropna(),color='steelblue')
sns.distplot(8.743+0.462*N2[D4000_mode==1].dropna(),color='firebrick')
sns.distplot(8.743+0.462*N2[D4000_mode==-1].dropna(),color='steelblue')



