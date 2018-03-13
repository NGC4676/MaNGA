import numpy as np
from astropy.io import ascii, fits
from astropy.table import Table, Column
from scipy.ndimage import gaussian_filter
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import shutil
import seaborn as sns

NII_plot=np.linspace(-1.5,0.0,100)
def BPT_border(log_NII_Ha,linetype = "Kauffmann"):
	if linetype=="Kauffmann": return 1.3 + 0.61 / (log_NII_Ha-0.05)
	elif linetype=="Kewley": return 1.19 + 0.61 / (log_NII_Ha - 0.47)

def BPT_distance(bpt_x,bpt_y,NII=NII_plot):
	OIII = BPT_border(NII)
	dists = np.sqrt((bpt_x-NII)**2+(bpt_y-OIII)**2)
	dist = np.min(dists)
	if (bpt_y < BPT_border(bpt_x)) & (bpt_x<NII_plot.max()): return dist
	else: return -1*dist

def BPT_plot(NII_Ha,OIII_Hb):
	plt.figure()
	NII_plot=np.linspace(-1.5,0.0,100)
	NII_plot2=np.linspace(-1.5,0.45,100)
	plt.plot(NII_plot,BPT_border(NII_plot, 'Kauffmann'),c='k')
	plt.plot(NII_plot2,BPT_border(NII_plot2, 'Kewley'),c='k',ls='--')
	SF_cond = (OIII_Hb<BPT_border(NII_Ha,'Kauffmann')) & (NII_Ha<0.)
	Composite_cond = (OIII_Hb>BPT_border(NII_Ha,'Kauffmann')) & (OIII_Hb<BPT_border(NII_Ha,'Kewley')) & (NII_Ha<0.45)
	plt.scatter(NII_Ha[SF_cond], OIII_Hb[SF_cond],
		c="steelblue",alpha=0.7)
	plt.scatter(NII_Ha[Composite_cond], OIII_Hb[Composite_cond],
		c="g",alpha=0.5)
	plt.scatter(NII_Ha[~((SF_cond)|(Composite_cond))], OIII_Hb[~((SF_cond)|(Composite_cond))],
                c="firebrick",alpha=0.7)
	plt.text(-1.25,-0.6,"%d"%len(NII_Ha[SF_cond]),fontsize=15,color="steelblue")
	plt.text(-0.15,-0.8,"%d"%len(NII_Ha[Composite_cond]),fontsize=15,color="g")
	plt.text(0.1,1.25,"%d"%len(NII_Ha[~((SF_cond)|(Composite_cond))]),fontsize=15,color="firebrick")
	plt.xlim(-1.5,0.5); plt.ylim(-1,1.5)

#--------Sample Selection--------#
Table_drp = Table.read('drpall-v2_1_2.fits')
table_drp = Table(Table_drp.columns["plateifu","mngtarg1","nsa_sersic_n","nsa_sersic_ba","nsa_sersic_th50","ifudesignsize"]).to_pandas()
table_drp['NUV_r'] = Table_drp.columns['nsa_sersic_absmag'][:,1] - Table_drp.columns['nsa_sersic_absmag'][:,4]

# Build pure SFG sample
table_pip3D = Table.read('manga.Pipe3D-v2_1_2.fits').to_pandas()

print "Merging catalogs... MaNGA DRP"
table_merg = pd.merge(table_pip3D,table_drp)
table_face = table_merg[table_merg.nsa_sersic_ba>0.5]    # face-on (b/a > 0.5)

# Classification from D4000
table_D4000 = ascii.read('beta_profile_D4000.dat').to_pandas()
data_sample = pd.merge(table_face,table_D4000)

# Use Flag from image (Remove merger, center-shifted)
table_use = ascii.read('beta_use_image.dat').to_pandas()
data_sample = pd.merge(table_use[table_use.flag==1], data_sample)


maps = dict({'127':76,'91':64,'61':54,'37':42,'19':34})
data_sample["fov_pix"] = [maps.get(str(ifu)) for ifu in data_sample["ifudesignsize"]]
data_sample["fov_re"] = data_sample.fov_pix/2.*0.5/data_sample.re_arc

print "Computing BPT distance..."
#BPT_x,BPT_y = data_sample.log_nii_ha_cen, data_sample.log_oiii_hb_cen
#data_sample['bpt_dist'] = [BPT_distance(bpt_x,bpt_y,NII=NII_plot) for (bpt_x,bpt_y) in zip(BPT_x,BPT_y)]

Weights = fits.open('weight.fits')[0].data.astype('float64')
Table_mpl5 = Table.read('mpl5_cat.fits')
MPL_plateifu = Table_mpl5['PLATEIFU'].astype('object')
MPL_ID = [ID.strip() for ID in MPL_plateifu]
Table_weights = pd.DataFrame({'plateifu':MPL_ID,'weight':Weights})
data_weight = pd.merge(Table_weights, data_sample)


# ----Analysis-----#
table_all = data_sample[data_sample.D4000_mode!=0]
table_io = data_sample[data_sample.D4000_mode==1]
table_oi = data_sample[data_sample.D4000_mode==-1]

table_all_w = data_weight[data_weight.D4000_mode!=0]
table_io_w = data_weight[data_weight.D4000_mode==1]
table_oi_w = data_weight[data_weight.D4000_mode==-1]

# PSF
table_all=table_all[table_all.re_arc>2.5]
table_io=table_io[table_io.re_arc>2.5]
table_oi=table_oi[table_oi.re_arc>2.5]
table_all_w=table_all_w[table_all_w.re_arc>2.5]
table_io_w=table_io_w[table_io_w.re_arc>2.5]
table_oi_w=table_oi_w[table_oi_w.re_arc>2.5]

cof_fit_io = np.polyfit(table_io.log_mass,table_io.log_sfr_ha,1)
cof_fit_oi = np.polyfit(table_oi.log_mass,table_oi.log_sfr_ha,1)
cof_fit_io_w = np.polyfit(table_io_w.log_mass,table_io_w.log_sfr_ha,1, w=table_io_w.weight)
cof_fit_oi_w = np.polyfit(table_oi_w.log_mass,table_oi_w.log_sfr_ha,1, w=table_oi_w.weight)

# ----SFMS Plotting
figure(figsize=(7,6))
gs = mpl.gridspec.GridSpec(2, 2, height_ratios=[1, 4],width_ratios=[4, 1])

ax2 = plt.subplot(gs[2])
bg = table_merg#[table_merg.NUV_r<4]
#sns.kdeplot(bg.log_mass, bg.log_sfr_ha, cmap="Greys_r",alpha=0.5,n_levels=10,zorder=1)

#---
H, xbins, ybins = np.histogram2d(bg.log_mass, bg.log_sfr_ha,
	                 bins=(np.linspace(8.5,11.5, 40),
	                       np.linspace(-2.6,1.65, 40)))
ax2.imshow(np.log10(H).T, cmap='Greys', origin='lower',
  		extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]],
  		interpolation='nearest', aspect='auto',alpha=0.7)
#---

#H, xbins, ybins = np.histogram2d(bg.log_mass, bg.log_sfr_ha, bins=(np.linspace(8., 12., 25), np.linspace(-2.5, 2., 30)))
#XH = np.sort(pd.Series(H[H!=0].ravel()))
#Hsum = XH.sum()
#XH_levels = [np.argmin(abs(np.cumsum(XH)-q*Hsum)) for q in [0.2,0.4,0.6,0.8]]
#levels = [XH[k] for k in XH_levels]
#plt.contour(gaussian_filter(H, sigma=.8, order=0).T, levels, zorder=1,alpha=0.6,extent=[xbins.min(),xbins.max(),ybins.min(),ybins.max()],linewidths=1,cmap='jet',linestyles='solid')

plt.scatter(table_io.log_mass,table_io.log_sfr_ha, label="Inside-out",
		s=20,c="r",lw=.5,edgecolors='k',alpha=.9,zorder=3)
plt.scatter(table_oi.log_mass,table_oi.log_sfr_ha, label="Outside-in",
		s=20,c="b",lw=.5,edgecolors='k',alpha=.9,zorder=3)

#plt.scatter(data_sample[Differ>500].log_mass, data_sample[Differ>500].log_sfr_ha,s=20, facecolors='none', edgecolors='r',alpha=0.7,label="Cum Inside-out")
#plt.scatter(data_sample[Differ<-500].log_mass, data_sample[Differ<-500].log_sfr_ha,s=20, facecolors='none', edgecolors='b',alpha=0.7,label="Cum Outside-in")

xplot = np.linspace(8,12.,100)
plt.plot(xplot,cof_fit_io[0]*xplot+cof_fit_io[1],c="firebrick",lw=4,alpha=0.9,zorder=2)
plt.plot(xplot,cof_fit_oi[0]*xplot+cof_fit_oi[1],c="mediumblue",lw=4,alpha=0.9,zorder=2)
#plt.plot(xplot,cof_fit_io_w[0]*xplot+cof_fit_io_w[1],c="firebrick",lw=4,ls='--',alpha=0.9,zorder=2)
#plt.plot(xplot,cof_fit_oi_w[0]*xplot+cof_fit_oi_w[1],c="mediumblue",lw=4,ls='--',alpha=0.9,zorder=2)

plt.ylabel(r'$\rm{log{\ }SFR[M_{\odot}yr^{-1}]}$',fontsize='large')
plt.xlabel(r'$\rm{log{\ }M_*[M_{\odot}]}$',fontsize='large')
plt.legend(loc="best",fontsize=12)
plt.xlim(8.5,11.5); plt.ylim(-2.6,1.65)

ax1 = plt.subplot(gs[0])
sns.distplot(table_oi.log_mass, color='royalblue')
sns.distplot(table_io.log_mass, color='firebrick')
ax1.set_xlim(ax2.get_xlim()); ax1.set_xlabel('')
plt.setp(ax1.get_xticklabels(), visible=False)
plt.setp(ax1.get_yticklabels(), visible=False)

ax3 = plt.subplot(gs[3])
sns.distplot(table_io.log_sfr_ha, color='firebrick',vertical=True)
sns.distplot(table_oi.log_sfr_ha, color='royalblue',vertical=True)
ax3.set_ylim(ax2.get_ylim()); ax3.set_ylabel('')
plt.setp(ax3.get_xticklabels(), visible=False)
plt.setp(ax3.get_yticklabels(), visible=False)

plt.subplots_adjust(left=0.12, bottom=0.1, right=0.95, top=0.95, wspace=0.001, hspace=0.001)

# Re Histogram
figure(figsize=(7,6))
sns.distplot(table_all.re_arc,bins=20,kde=False,norm_hist=False,color='grey')
axvline(2.5,ls="--",lw=3,color="r",alpha=0.7)
axvline(5.,ls="--",lw=3,color="mediumblue",alpha=0.7)
#frac1 = len(table_all[table_all.re_arc<2.5])/1.0/len(table_all)
#frac2 = len(table_all[(table_all.re_arc>2.5)&(table_all.re_arc<5.)])/1.0/len(table_all)
#frac3 = len(table_all[table_all.re_arc>5.])/1.0/len(table_all)
plt.text(1.8,5,'%d %%'%(5),fontsize=12,color='r')
plt.text(3.4,5,'%d %%'%(39),fontsize=12,color='g')
plt.text(6.5,5,'%d %%'%(56),fontsize=12,color='b')
plt.ylabel('Number',fontsize='large')
plt.xlabel(r'$\rm{R_e{\ }[arcsec]}$',fontsize='large')
plt.savefig("../Pub/Re_hist.pdf")

# Move mass growth curve (re-classification)
move_classify = False
if move_classify:
	for ID in table_cum_oi.plateifu:
		plate,ifu = re.findall(r'\d+?\d*e?\d*?',ID)
		pic = "/home/qliu/MaNGA/mass_growth/All/%s.png"%ID
		shutil.copy2(pic,"/home/qliu/MaNGA/mass_growth/Outside-in/")

	for ID in table_cum_io.plateifu:
		plate,ifu = re.findall(r'\d+?\d*e?\d*?',ID)
		pic = "/home/qliu/MaNGA/mass_growth/All/%s.png"%ID
		shutil.copy2(pic,"/home/qliu/MaNGA/mass_growth/Inside-out/")

