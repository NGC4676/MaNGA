import numpy as np
import pandas as pd
from astropy.io import fits, ascii
from astropy.table import Table, Column
import matplotlib.pyplot as plt
from statstool import leastsq, pearson
from scipy.stats import gaussian_kde, pearsonr
from scipy.ndimage import gaussian_filter
from astroML.linear_model import LinearRegression

def median_fitting(x,y,q=0.025,d=8):
	xq = np.linspace(x.quantile(q),x.quantile(1-q),d+1)
	xp = [(xq[k]+xq[k+1])/2. for k in range(d)]
	yp = [np.median(y[(x>xq[k])&(x<xq[k+1])]) for k in range(d)]
	cof = np.polyfit(xp,yp,1)
	return xp,yp,cof

def mode_fitting(x,y,q=0.01,d=8):
	xq = np.linspace(x.quantile(q),x.quantile(1-q),d+1)
	xp = [(xq[k]+xq[k+1])/2. for k in range(d)]
	ypos = np.linspace(y.quantile(q),y.quantile(1-q),50)
	yp = np.empty_like(xp)
	for k in range(d):
		yvals = y[(x>xq[k])&(x<xq[k+1])]
		ind = np.argmax(gaussian_kde(yvals).evaluate(ypos))
		yp[k] = ypos[ind]
	cof = np.polyfit(xp,yp,1)
	return xp,yp,cof

#--------Merging Measurements--------#

# Read Spaxel Values
#table = ascii.read('/home/qliu/MaNGA/beta_val_kew2.dat')
table = ascii.read('/home/qliu/MaNGA/delta_val_kew.dat')
data = table.to_pandas()
data_SF = data[(data.bpt!=0)&(data.ew_ha_mean<-6)]

# Classification from vote for Mass Growth CDF
table_vote = ascii.read('/home/qliu/MaNGA/beta_vote_SFG.dat').to_pandas()
data_SF = pd.merge(table_vote, data_SF)

# Use Flag from image (Remove merger, center-shifted)
table_use = ascii.read('/home/qliu/MaNGA/beta_use_image.dat').to_pandas()
data_SF = pd.merge(table_use[table_use.flag==1],data_SF)

print "Reading Pipe3D Table..."
Table_pip3D = Table.read('/home/qliu/MaNGA/pipe3d/manga.Pipe3D-v2_1_2.fits')
table_pip3D = Table_pip3D.to_pandas()

data_SF = pd.merge(table_pip3D,data_SF) 

data_SF['Sigma_mol'] = np.log10(15*data_SF.Av + (8.743+0.462*data_SF.N2 -12 -2.67))

P0 = data_SF[data_SF.mode_flag!=0]
P0 = P0[P0.re_arc>2.5]
P1, P2 = P0[P0.mode_flag==1], P0[P0.mode_flag==-1]
xplot=np.linspace(4.5,11.5,100)

Weights = fits.open('/home/qliu/MaNGA/weight.fits')[0].data.astype('float64')
Table_mpl5 = Table.read('/home/qliu/MaNGA/mpl5_cat.fits')
MPL_plateifu = Table_mpl5['PLATEIFU'].astype('object')
MPL_ID = [ID.strip() for ID in MPL_plateifu]
Table_weights = pd.DataFrame({'plateifu':MPL_ID,'weight':Weights})
data_SF_w = pd.merge(Table_weights, data_SF)
P0_w = data_SF_w[data_SF_w.mode_flag!=0]
P0_w = P0_w[P0_w.re_arc>2.5]
P1_w, P2_w = P0_w[P0_w.mode_flag==1], P0_w[P0_w.mode_flag==-1]

#--------Merging Measurements Over--------#

# Use 80% data to fit
W = data_SF[data_SF.mode_flag!=0]
xx, yy = np.mgrid[5:11:60j, -5:1:60j]
positions = np.vstack([xx.ravel(), yy.ravel()])
W0 = W.sample(frac=0.1)
kernel = gaussian_kde(np.vstack([W0.Sigma_Mass, W0.Sigma_SFR]))
print "Use KDE to derive PDF... %d points used."%len(W0)
pdf = pd.Series(kernel.pdf(np.vstack([W.Sigma_Mass, W.Sigma_SFR])))
use_80 = (pdf > pdf.quantile(0.2)).values
W_80 = W[use_80]
print "Select 80% of data... Finish!"

# plot Fig.2
mp,sfrp,cof = median_fitting(P0.Sigma_Mass, P0.Sigma_SFR, q=0.01,d=7)
m_i,sfr_i,cof_i = median_fitting(P1.Sigma_Mass, P1.Sigma_SFR, q=0.01,d=7)
m_o,sfr_o,cof_o = median_fitting(P2.Sigma_Mass, P2.Sigma_SFR, q=0.01,d=7)

plt.figure(figsize=(15,5.))
for i, (W,c,cmap,lab,p) in enumerate(zip([P0,P1,P2],['k','r','b'],['Greys','Reds','Blues'],['Total','Inside-out','Outside-in'],['a','b','c'])):
	ax = plt.subplot(1,3,i+1)
	plt.text(0.018,0.05,"%s)"%p,fontsize='large',transform=ax.transAxes)
	H, xbins, ybins = np.histogram2d(W.Sigma_Mass, W.Sigma_SFR,
		                 bins=(np.linspace(5.0, 11., 50),
		                       np.linspace(-5., 1., 50)))
	ax.imshow(np.log10(H).T, cmap=cmap, origin='lower',
	  		extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]],
	  		interpolation='nearest', aspect='equal',alpha=0.7)

	H, xbins, ybins = np.histogram2d(W.Sigma_Mass, W.Sigma_SFR,
				bins=(np.linspace(5.0, 11., 50), np.linspace(-5., 1., 50)))
	XH = np.sort(pd.Series(H[H!=0].ravel()))
	Hsum = XH.sum()
	XH_levels = [np.argmin(abs(np.cumsum(XH)-q*Hsum)) for q in [0.01,0.05,0.32,0.7]]
	levels = [XH[k] for k in XH_levels]

	plt.contour(gaussian_filter(H, sigma=.8, order=0).T, levels, extent=[xbins.min(),xbins.max(),
	    ybins.min(),ybins.max()],linewidths=1,colors='black',linestyles='solid')

	#p_d = leastsq(W.Sigma_Mass, W.Sigma_SFR,method=4)
	#plt.plot(xplot, p_d[1]*xplot+p_d[0], c=c, ls='--',lw=3,label='ODS',alpha=0.7)
	
	plt.plot(xplot, 0.72*xplot-7.95, 'g--',lw=3,label='C16',alpha=0.7)
	plt.plot(xplot, 1.*xplot-10.33, 'm:',lw=4,label='H17',alpha=0.7)

	#p_w0 = np.polyfit(W.Sigma_Mass, W.Sigma_SFR,1,w=W.weight)
	p_b0 = leastsq(W.Sigma_Mass, W.Sigma_SFR,method=1)
	print 'beta，alpha，sigma_alpha:',p_b0
	plt.plot(xplot, xplot * p_b0[1] + p_b0[0],c=c, ls='-',lw=4,alpha=1.,label=lab)
	#plt.plot(xplot, xplot * p_w0[0] + p_w0[1],c=c, ls='--',lw=4,alpha=1.,label=lab+' (wc)')

	std = sqrt(np.sum((W.Sigma_Mass*p_b0[1]+p_b0[0]-W.Sigma_SFR)**2) / len(W.Sigma_Mass))
	print 'scatter:%.3f'%std

	if i==0:
		plt.plot(mp, sfrp, c='w',ls='-',marker='s',ms=7, lw=1,alpha=1.)
		plt.ylabel(r'$\rm{log_{10}(\Sigma_{SFR}[M_{\odot}yr^{-1}kpc^{-2}])}$',fontsize='large')
	else: yticklabels = ax.get_yticklabels(); plt.setp(yticklabels, visible=False)
	if i==1: plt.plot(m_i,sfr_i,c='w',ls='-',marker='s',ms=7, lw=1,alpha=1.)
	if i==2: plt.plot(m_o,sfr_o,c='w',ls='-',marker='s',ms=7, lw=1,alpha=1.)
	
	text(0.7,0.1,r'$\alpha=%.2f$'%(p_b0[1]),transform=ax.transAxes,fontsize=12)
	text(0.7,0.05,r'$\beta=%.2f$'%(p_b0[0]),transform=ax.transAxes,fontsize=12)

	#text(0.55,0.1,r'$\alpha=%.2f$'%(p_b0[1]),transform=ax.transAxes,fontsize=12)
	#text(0.55,0.05,r'$\beta=%.2f$'%(p_b0[0]),transform=ax.transAxes,fontsize=12) 
	#text(0.8,0.1,r'$(%.2f)$'%(p_w0[0]),transform=ax.transAxes,fontsize=12)
	#text(0.8,0.05,r'$(%.2f)$'%(p_w0[1]),transform=ax.transAxes,fontsize=12)

	#text(0.7,0.2,r'$\alpha_{ODS}=%.2f$'%(p_d[1]),transform=ax.transAxes,fontsize=12)
	#text(0.7,0.15,r'$\beta_{ODS}=%.2f$'%(p_d[0]),transform=ax.transAxes,fontsize=12)

	xlabel(r'$\rm{log_{10}(\Sigma_{*}[M_{\odot}kpc^{-2}])}$',fontsize='large')
	xlim(5.01, 10.99); plt.ylim(-5.,1.)
	legend(fontsize=12,frameon=True,loc=2)
subplots_adjust(left=0.075,bottom=0.1,right=0.95,top=0.95,wspace=0.0001,hspace=0.001)


### KS Law? ###
mp,sfrp,cof = median_fitting(P0.Sigma_Mass, P0.Sigma_mol, q=0.01,d=7)
m_i,sfr_i,cof_i = median_fitting(P1.Sigma_Mass, P1.Sigma_mol, q=0.01,d=7)
m_o,sfr_o,cof_o = median_fitting(P2.Sigma_Mass, P2.Sigma_mol, q=0.01,d=7)

plt.figure(figsize=(15,5.))
for i, (W,c,cmap,lab,p) in enumerate(zip([P0,P1,P2],['k','r','b'],['Greys','Reds','Blues'],['Total','Inside-out','Outside-in'],['a','b','c'])):
	ax = plt.subplot(1,3,i+1)
	plt.text(0.018,0.05,"%s)"%p,fontsize='large',transform=ax.transAxes)
	H, xbins, ybins = np.histogram2d(W.Sigma_Mass, W.Sigma_mol,
		                 bins=(np.linspace(5.0, 11., 50),
		                       np.linspace(-4., 2., 50)))
	ax.imshow(np.log10(H).T, cmap=cmap, origin='lower',
	  		extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]],
	  		interpolation='nearest', aspect='equal',alpha=0.7)

	H, xbins, ybins = np.histogram2d(W.Sigma_Mass, W.Sigma_mol,
				bins=(np.linspace(5.0, 11., 50), np.linspace(-4., 2., 50)))
	XH = np.sort(pd.Series(H[H!=0].ravel()))
	Hsum = XH.sum()
	XH_levels = [np.argmin(abs(np.cumsum(XH)-q*Hsum)) for q in [0.01,0.05,0.32,0.7]]
	levels = [XH[k] for k in XH_levels]

	plt.contour(gaussian_filter(H, sigma=.8, order=0).T, levels, extent=[xbins.min(),xbins.max(),
	    ybins.min(),ybins.max()],linewidths=1,colors='black',linestyles='solid')


	#p_w0 = np.polyfit(W.Sigma_Mass, W.Sigma_SFR,1,w=W.weight)
	p_b0 = leastsq(W.Sigma_Mass, W.Sigma_mol,method=1)
	print 'beta，alpha，sigma_alpha:',p_b0
	plt.plot(xplot, xplot * p_b0[1] + p_b0[0],c=c, ls='-',lw=4,alpha=1.,label=lab)


	std = sqrt(np.sum((W.Sigma_Mass*p_b0[1]+p_b0[0]-W.Sigma_mol)**2) / len(W.Sigma_Mass))
	print 'scatter:%.3f'%std

	if i==0:
		plt.plot(mp, sfrp, c='w',ls='-',marker='s',ms=7, lw=1,alpha=1.)
		plt.ylabel(r'$\rm{log_{10}(\Sigma_{SFR}[M_{\odot}yr^{-1}kpc^{-2}])}$',fontsize='large')
	else: yticklabels = ax.get_yticklabels(); plt.setp(yticklabels, visible=False)
	if i==1: plt.plot(m_i,sfr_i,c='w',ls='-',marker='s',ms=7, lw=1,alpha=1.)
	if i==2: plt.plot(m_o,sfr_o,c='w',ls='-',marker='s',ms=7, lw=1,alpha=1.)

	xlabel(r'$\rm{log_{10}(\Sigma_{mol}[M_{\odot}pc^{-2}])}$',fontsize='large')
	xlim(5., 11.); plt.ylim(-4.,2.)
	legend(fontsize=12,frameon=True,loc=2)
subplots_adjust(left=0.075,bottom=0.1,right=0.95,top=0.95,wspace=0.0001,hspace=0.001)

 
