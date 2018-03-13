import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from scipy.stats import ks_2samp,spearmanr,kendalltau
from astropy.io import fits, ascii
from astropy.table import Table, Column
import seaborn as sns
import shutil
import re

def median_fitting(x,y,q=0.03,d=8):
	xq = np.linspace(x.quantile(q),x.quantile(1-q),d+1)
	xp = [xq[k] for k in range(d+1)]
	yp = [np.median(y[(x>xq[k]-x.std())&(x<xq[k]+x.std())]) for k in range(d+1)]
	cof = np.polyfit(xp,yp,1)
	return xp,yp,cof

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
table_drp = Table(Table_drp.columns["plateifu","nsa_sersic_n","nsa_sersic_ba"]).to_pandas()
table_drp['NUV_r'] = Table_drp.columns['nsa_sersic_absmag'][:,1] - Table_drp.columns['nsa_sersic_absmag'][:,4]

print "Reading Pipe3D Table..."
Table_pip3D = Table.read('manga.Pipe3D-v2_1_2.fits')
table_pip3D = Table_pip3D.to_pandas()

print "Merging catalogs... MaNGA DRP"
table_face = pd.merge(table_pip3D,table_drp[table_drp.nsa_sersic_ba>0.5])    # face-on (b/a > 0.5)

#--------Merging Measurements--------#

# SGMS Fitting (Slope,ZP)
table_fit = ascii.read('delta_fit_SGMS_kew.dat').to_pandas()
data_fit = pd.merge(table_fit, table_face)

# Morphology Paras from EW(Ha)
table_morph = ascii.read('beta_stats_morph_SFG.dat').to_pandas()
data_fit = pd.merge(data_fit, table_morph)

# Classification from D4000
table_D4000 = ascii.read('beta_profile_D4000.dat').to_pandas()
data_fit = pd.merge(data_fit, table_D4000)

# Classification from vote for Mass Growth CDF
#table_vote = ascii.read('/home/qliu/MaNGA/beta_vote_SFG.dat').to_pandas()
#data_fit = pd.merge(table_vote, data_fit)

# Use Flag from image (Remove merger, center-shifted)
table_use = ascii.read('beta_use_image.dat').to_pandas()
data_fit = pd.merge(table_use[table_use.flag==1], data_fit)
data_fit = data_fit[data_fit.re_arc>2.5]

# AGN List from Sanchez 2017
#table_AGN=pd.read_table("AGN_Sanchez17.dat")

# Use B/T from Simard 2011
table_simard = Table.read('match_simard11.fits').to_pandas()

Table_mpl5 = Table.read('mpl5_cat.fits')
MPL_plateifu = Table_mpl5['PLATEIFU'].astype('object')
MPL_ID = [ID.strip() for ID in MPL_plateifu]
table_BT = pd.DataFrame({'plateifu':MPL_ID,
			'BT_R':table_simard.BT_R, 'BT_G':table_simard.BT_G})
data_fit = pd.merge(table_BT, data_fit)

#--------Merging Measurements Over--------#


# Compute new params
data_fit['log_ssfr'] = data_fit.log_sfr_ha - data_fit.log_mass
data_fit['gas_ratio'] = 10**data_fit.log_mass_gas/(10**data_fit.log_mass_gas + 10**data_fit.log_mass)
BPT_x,BPT_y = data_fit.log_nii_ha_cen, data_fit.log_oiii_hb_cen
#data_fit['bpt_dist'] = [BPT_distance(bpt_x,bpt_y,NII=NII_plot) for (bpt_x,bpt_y) in zip(BPT_x,BPT_y)]

# Subsample
table_fit_io = data_fit[data_fit.D4000_mode==1]
table_fit_oi = data_fit[data_fit.D4000_mode==-1]

#BPT_plot(data_fit.log_nii_ha_cen, data_fit.log_oiii_hb_cen)
#BPT_plot(table_fit_io.log_nii_ha_cen, table_fit_io.log_oiii_hb_cen)
#BPT_plot(table_fit_oi.log_nii_ha_cen, table_fit_oi.log_oiii_hb_cen)


# Plot
def Scatter_KS(S1, S2, atts, xlabels=None, code='log_ssfr',Nq=[4,4,4],q=0.05,Textlocs=['left','left','right']):
	S = pd.concat([S1,S2])
	#S1_cdf = S[S.weight_difference>500]
	#S2_cdf = S[S.weight_difference<-500]

	fig = plt.figure(figsize=(16,5))
	gs = mpl.gridspec.GridSpec(2, 4, height_ratios=[3, 1],width_ratios=[3,3,3,1])
	for i,(att,xlab,p,nq,textloc) in enumerate(zip(atts,xlabels,['a','b','c'],Nq,Textlocs)):
		#scatter
		ax1 = plt.subplot(gs[i])
		plt.text(0.05,0.9,"%s)"%p,fontsize='large',transform=ax1.transAxes)

		#xplot = np.linspace(data_fit[att].min()-2.,data_fit[att].max()+2.,20)
		#p1 = np.polyfit(S1[att], S1.orslope, 1)
		#p2 = np.polyfit(S2[att], S2.orslope, 1)
		#sns.regplot(S1[att], S1.orslope,ci=90,color='firebrick',
             #       scatter=True,robust=True,line_kws={'alpha':0.7})
		#sns.regplot(S2[att], S2.orslope,ci=90,color='mediumblue',
             #       scatter=True,robust=True,line_kws={'alpha':0.7})

		s1 = scatter(S1[att],S1.orslope,s=50,lw=0,c='darkred',alpha=0.7,zorder=1)
		s2 = scatter(S2[att],S2.orslope,s=50,lw=0,c='mediumblue',alpha=0.7,zorder=1)

		#if nq==0:
			#a1p,s1p,cof1 = median_fitting(S1[att],S1.orslope,q=q,d=5)
			#plt.plot(a1p, s1p,c='r', ls='-',lw=6,alpha=1.,zorder=2)
		if nq!=0:
			a1p,s1p,cof1 = median_fitting(S1[att],S1.orslope,q=q,d=nq)
			a2p,s2p,cof2 = median_fitting(S2[att],S2.orslope,q=q,d=nq)
			plt.plot(a1p, s1p,c='r', ls='-',lw=6,alpha=1.,zorder=2)
			plt.plot(a2p, s2p,c='deepskyblue',ls='-',lw=6,alpha=1.,zorder=2)

		plt.axhline(1.04,c='k',ls='--',alpha=0.7,zorder=2)
		if i == 0: plt.ylabel('Slope of G-by-G SGMS',fontsize='large')
		else: plt.setp(ax1.get_yticklabels(), visible=False);
		plt.setp(ax1.get_xticklabels(), visible=False)

		# histogram att
		ax2 = plt.subplot(gs[i+4])
		sns.distplot(S1[att].dropna(),color='firebrick')
		sns.distplot(S2[att].dropna(),color='royalblue')
		D_stat,p_value = ks_2samp(S1[att].dropna(), S2[att].dropna())
		
		plt.xlabel(xlab,fontsize='large')
		
		if textloc=="right": 
			plt.text(0.8,0.6,'D: %.2f\np: %.1e'%(D_stat,p_value), transform=ax2.transAxes)
		
		else: plt.text(0.025,0.6,'D: %.2f\np: %.1e'%(D_stat,p_value), transform=ax2.transAxes)

		ax2.set_xlim(ax1.get_xlim())
		plt.setp(ax2.get_yticklabels(), visible=False)	

	# histogram slope
	ax3 = plt.subplot(gs[3])
	sns.distplot(S1.orslope.dropna(),color='firebrick',vertical=True)
	sns.distplot(S2.orslope.dropna(),color='royalblue',vertical=True)

	D_stat,p_value = ks_2samp(S1.orslope.dropna(), S2.orslope.dropna())

	axhline(1.04,c='k',ls='--',alpha=0.7)
	axhline(y=1.26,xmin=0,xmax=.63,c='b',ls='--',alpha=0.6)
	axhline(y=.7,xmin=0,xmax=.88,c='r',ls='--',alpha=0.6)
	text(-0.5,1.225,'1.25',fontsize=10,color='b',alpha=0.6)
	text(-0.375,0.675,'0.7',fontsize=10,color='r',alpha=0.6)
	text(0.2,0.85,'D: %.2f\np: %.1e'%(D_stat,p_value), transform=ax3.transAxes)
	ax3.set_ylim(ax1.get_ylim()); plt.setp(ax3.get_xticklabels(), visible=False)
	ylabel(''); setp(ax3.get_yticklabels(), visible=False)
	subplots_adjust(left=0.05, bottom=0.15, right=0.975, top=0.9, wspace=0.001, hspace=0.001)
	return gs

# Global SSP Property
S1, S2 = table_fit_io[table_fit_io.r2>0.6], table_fit_oi[table_fit_oi.r2>0.6]


atts1 = ['log_mass','log_ssfr','bpt_dist']
xlabels=[r'$\rm{log_{10}(M_*[M_{\odot}])}$',r'$\rm{log_{10}(sSFR[yr^{-1}])}$','BPT Distance']
gs = Scatter_KS(S1, S2, atts1,xlabels=xlabels,Nq=[4,4,0],q=0.03,Textlocs=['left','left','left'])
ax = plt.subplot(gs[0]); ax.set_xlim(8.75,11.45)
axh = plt.subplot(gs[4]); axh.set_xlim(ax.get_xlim())
ax = plt.subplot(gs[1]); ax.set_xlim(-11.225,-9.55)
p,cov=np.polyfit(S1.log_ssfr, S1.orslope,1,cov=True)
xplot=linspace(S1.log_ssfr.min(),S1.log_ssfr.max(),20)
b_err=sqrt(cov[1,1])
plot(xplot,xplot*p[0]+p[1],ls="--",lw=3,color="gold",alpha=.8) 
axh = plt.subplot(gs[5]); axh.set_xlim(ax.get_xlim())
ax = plt.subplot(gs[2]); ax.set_xlim(-0.88,0.65)
text(-0.58,2.4,r"$\bf AGN$",color="orange",fontsize=12)
text(0.4,2.4,r"$\bf SF$",color="steelblue",fontsize=12)
ax.axvline(0.,color='orange',ls=':',lw=3)
ax.annotate("", xy=(-0.35, 2.45), xytext=(-0.1, 2.45), arrowprops=dict(color="orange",width=2,headwidth=8),alpha=0.7)
ax.annotate("", xy=(0.35, 2.45), xytext=(0.1, 2.45), arrowprops=dict(color="steelblue",width=2,headwidth=8),alpha=0.7)
axh = plt.subplot(gs[6]); axh.set_xlim(ax.get_xlim())


atts1 = ['log_mass','log_ssfr','alpha_age_lw_re_fit']
xlabels=[r'$\rm{log{\ }M_*[M_{\odot}]}$',r'$\rm{log{\ }sSFR[yr^{-1}]}$',r'$\rm{\alpha\langle Age_{LW} \rangle}$']
gs = Scatter_KS(S1, S2, atts1,xlabels=xlabels,Nq=[4,4,4],q=0.03,Textlocs=['left','left','left'])
ax = plt.subplot(gs[0]); ax.set_xlim(8.75,11.45)
axh = plt.subplot(gs[4]); axh.set_xlim(ax.get_xlim())
ax = plt.subplot(gs[1]); ax.set_xlim(-11.225,-9.55)
#p,cov=np.polyfit(S1.log_ssfr, S1.orslope,1,cov=True)
xplot=linspace(S1.log_ssfr.min(),S1.log_ssfr.max(),20)
#plot(xplot,xplot*p[0]+p[1],ls="--",lw=3,color="gold",alpha=.8) 
axh = plt.subplot(gs[5]); axh.set_xlim(ax.get_xlim())



atts2 = ['alpha_oh_re_fit_o3n2','alpha_zh_lw_re_fit','alpha_age_lw_re_fit']
xlabels=[r'$\rm{\alpha\langle{(O/H)}_{O3N2}\rangle}$',r'$\rm{\alpha\langle{(Z/H)}_{LW}\rangle}$',r'$\rm{\alpha\langle Age_{LW} \rangle}$']
gs = Scatter_KS(S1, S2, atts2,xlabels=xlabels, Nq=[4,4,4],q=0.01,Textlocs=['left','left','left'])
ax = plt.subplot(gs[0]); ax.set_xlim(-0.22,0.14)
axh = plt.subplot(gs[4]); axh.set_xlim(ax.get_xlim()); axh.set_ylim(0,9.5)
axh = plt.subplot(gs[5]); axh.set_ylim(0,8.5)


atts3 = ['nsa_sersic_n','BT_G','lambda_re']
xlabels=[r'$\rm Sersic n$',r'$\rm (B/T)_G$',r'$\rm{\lambda_{R}(1.5{\ }R_e)}$']
gs = Scatter_KS(S1[S1.BT_G>-0.1], S2, atts3,xlabels=xlabels, Nq=[4,4,4],q=0.03,Textlocs=['right','right','right'])
ax = plt.subplot(gs[0]); ax.set_xlim(0.25,6.15)
axh = plt.subplot(gs[4]); axh.set_xlim(ax.get_xlim())
ax = plt.subplot(gs[1]); ax.set_xlim(-0.02,1.02)
axh = plt.subplot(gs[5]); axh.set_xlim(ax.get_xlim())
ax = plt.subplot(gs[2]);ax.set_xlim(0.02,1.02)
p,cov=np.polyfit(S1[S1.BT_G>-0.1].lambda_re,S1[S1.BT_G>-0.1].orslope,1,cov=True)
xplot=linspace(S1.lambda_re.min(),S1.lambda_re.max(),20)
b_err=sqrt(cov[1,1])
plot(xplot,xplot*p[0]+p[1],ls="--",lw=3,color="gold",alpha=1.) 
#fill_between(xplot,xplot*p[0]+p[1]-3*b_err,xplot*p[0]+p[1]+3*b_err,color="orangered",alpha=0.3,zorder=1)
axh = plt.subplot(gs[6]);axh.set_xlim(ax.get_xlim())


K1, K2 = S1[S1.morph_flag==0], S2[S2.morph_flag==0] 
atts4 = ['m20','A','C']
xlabels=['M20','A','C']
gs = Scatter_KS(K1, K2, atts4,xlabels=xlabels, Nq=[5,5,5],q=0.05)

# Quick Look
def median_fitting2(x,y,d=10):
	y = y[(x>0)&(x<5.)]
	x = x[(x>0)&(x<5.)]
	xq = np.linspace(x.min(),x.max(),d+1)
	xp = [(xq[k]) for k in range(d)]
	yp = np.array([y[(x>xq[k])&(x<xq[k+1])].median() for k in range(d)]).ravel()
	yp_a = np.array([y[(x>xq[k])&(x<xq[k+1])].quantile(0.2) for k in range(d)]).ravel()
	yp_b = np.array([y[(x>xq[k])&(x<xq[k+1])].quantile(0.8) for k in range(d)]).ravel()
	#cof = np.polyfit(xp,yp,1)
	return xp,yp,yp_a,yp_b


### Fig 6 ###
figure(figsize=(11,5))
gs = mpl.gridspec.GridSpec(1, 2,width_ratios=[3, 2])

ax1 = plt.subplot(gs[0])	
Ja, Jb = S1[S1.orslope<S1.orslope.quantile(0.25)], S1[S1.orslope>S1.orslope.quantile(0.75)]
scatter(Ja.bpt_dist,Ja.lambda_re,s=50,color='chocolate',label='slope<0.6',alpha=0.7) 
scatter(Jb.bpt_dist,Jb.lambda_re,s=50,color='green',label='slope>0.9',alpha=0.7) 
legend(loc='best',fontsize='large')
text(-0.85,S1.lambda_re.quantile(0.75)-0.05,r"$\%75{\}\lambda_R$",color="navy",fontsize=12)
text(-0.85,S1.lambda_re.quantile(0.25)-0.05,r"$\%25{\}\lambda_R$",color="darkred",fontsize=12)
axhline(S1.lambda_re.quantile(0.75),lw=2,ls=":",color='navy')  
axhline(S1.lambda_re.quantile(0.25),lw=2,ls=":",color='darkred')  
text(-0.58,0.1,r"$\bf AGN$",color="orange",fontsize=15)
text(0.4,0.1,r"$\bf SF$",color="steelblue",fontsize=15)
axvline(0.,color='orange',ls=':',lw=3)
annotate("", xy=(-0.35, 0.125), xytext=(-0.1, 0.125), arrowprops=dict(color="orange",width=2,headwidth=8),alpha=0.7)
annotate("", xy=(0.35, 0.125), xytext=(0.1, 0.125), arrowprops=dict(color="steelblue",width=2,headwidth=8),alpha=0.7)
xlabel('BPT Distance',fontsize='large')
ylabel(r'$\rm \lambda_R(1.5{\ }Re)$',fontsize='large')

ax2 = plt.subplot(gs[1])
sns.distplot(Ja.lambda_re, color='chocolate',label='slope<0.6')
sns.distplot(Jb.lambda_re, color='green',label='slope>0.9')
legend(loc='best',fontsize=12)
xlim(-0.1,1.1); xlabel(r'$\rm \lambda_R(1.5{\ }Re)$',fontsize='large')
tight_layout()

### Bulge ###
Ja = Ja[Ja.BT_G>-0.01]; Jb = Jb[Jb.BT_G>-0.01]
figure(figsize=(5,5))
sns.distplot(Ja.BT_G, color='chocolate',label='slope<0.6')
sns.distplot(Jb.BT_G, color='green',label='slope>0.9')
legend(loc='best',fontsize=12)
xlim(-0.02,1.02); xlabel(r'$\rm (B/T)_G$',fontsize='large')
tight_layout()


### gas ratio ###
figure(figsize=(11,5))
gs = mpl.gridspec.GridSpec(1, 2,width_ratios=[3, 2])

ax1 = plt.subplot(gs[0])	
att='gas_ratio'
s1 = plt.scatter(S1[att], S1.orslope, c='darkred',s=50,cmap='Reds',alpha=0.7)
s2 = plt.scatter(S2[att], S2.orslope, c='mediumblue',s=50,cmap='Blues',alpha=0.7)
rm1,tm1, cof = median_fitting(S1[att],S1.orslope,d=4,q=0.03)
rm2,tm2, cof = median_fitting(S2[att],S2.orslope,d=5,q=0.03)
plt.plot(rm1, tm1,c='r', ls='-',lw=10,alpha=1.,zorder=2)
plt.plot(rm2, tm2,c='deepskyblue',ls='-',lw=10,alpha=1.,zorder=2)
xlabel('Gas Fraction',fontsize='large')
ylabel('Slope of G-by-G SGMS',fontsize='large')

ax2 = plt.subplot(gs[1])
sns.distplot(S2[att], color='royalblue',label='Outside-in')
sns.distplot(S1[att], color='firebrick',label='Inside-out')
legend(loc='best',fontsize=12)
xlabel('Gas Fraction',fontsize='large')
tight_layout()


### Match Sample ###
M2 =  pd.DataFrame()
for (f,s2) in S2.iterrows():
	z2 = s2.redshift
	ind_zmin=argmin(abs(z2-S1.redshift))
	M2 = M2.append(S1[S1.index==ind_zmin])

figure(figsize=(9,4))
gs = mpl.gridspec.GridSpec(1, 2,width_ratios=[2, 2])
ax1 = plt.subplot(gs[0])
sns.distplot(S2.redshift,label='Outside-in')
sns.distplot(M2.redshift,label='Inside-out (matched)')
legend()
ax2 = plt.subplot(gs[1])
sns.distplot(S2.log_mass,label='Outside-in')
sns.distplot(M2.log_mass,label='Inside-out (matched)')
xlabel(r"$\rm M_*$",fontsize="large")
legend()
tight_layout()



# Move mass growth curve (re-classification)
move_pic = False
if move_pic:
	for ID in table_fit_oi.plateifu:
		plate,ifu = re.findall(r'\d+?\d*e?\d*?',ID)
		pic = "/home/qliu/MaNGA/SGMS_kew/All/%s.png"%ID
		try: shutil.copy2(pic,"/home/qliu/MaNGA/SGMS_kew/Outside-in/")
		except IOError: continue

	for ID in table_fit_io.plateifu:
		plate,ifu = re.findall(r'\d+?\d*e?\d*?',ID)
		pic = "/home/qliu/MaNGA/SGMS_kew/All/%s.png"%ID
		try: shutil.copy2(pic,"/home/qliu/MaNGA/SGMS_kew/Inside-out/")
		except IOError: continue
