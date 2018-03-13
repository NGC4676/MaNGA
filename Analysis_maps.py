import numpy as np
from astropy.io import fits, ascii
from astropy.table import Table, Column
from scipy import optimize, stats
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.patches as patches
import matplotlib.image as mpimg
import pandas as pd
import seaborn as sns
from astropy.stats import sigma_clip
import matplotlib as mpl
from astropy.modeling import models, fitting

model_ini = models.Linear1D(1.,0.)
fit = fitting.LevMarLSQFitter()
or_fit = fitting.FittingWithOutlierRemoval(fit, sigma_clip, niter=3, sigma=3.0)

xp = np.linspace(-2.5,2.5,100)

def median_fitting(x,y,q=0.025,d=8):
	xq = np.linspace(x.quantile(q),x.quantile(1-q),d+1)
	xp = [(xq[k]+xq[k+1])/2. for k in range(d)]
	yp = [np.median(y[(x>xq[k])&(x<xq[k+1])]) for k in range(d)]
	cof = np.polyfit(xp,yp,1)
	return xp,yp,cof

def mode_fitting(x,y,q=0.025,d=8):
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

def linear(x, k, b):
    return (k*x + b)

def piecewise_linear(x, x0, y0, k1, k2):
    return np.piecewise(x, [x < x0], [lambda x:k1*x + y0-k1*x0, lambda x:k2*x + y0-k2*x0])

def piecewise_linear_b(x, x0, y0, k1, k2):
    return np.piecewise(x, [x > x0], [lambda x:k1*x + y0-k1*x0, lambda x:k2*x + y0-k2*x0])

#--------Merging Measurements--------#
# Read Spaxel Values
table = ascii.read('delta_val_kew2.dat')
data = table.to_pandas()
data_SF = data[(data.bpt!=0)&(data.ew_ha_mean<-6)]

# DRP
Table_drp = Table.read('drpall-v2_1_2.fits')
table_drp = Table(Table_drp.columns["plateifu","mngtarg1","nsa_sersic_n","nsa_sersic_ba","ifudesignsize"]).to_pandas()
table_drp['NUV_r'] = Table_drp.columns['nsa_sersic_absmag'][:,1] - Table_drp.columns['nsa_sersic_absmag'][:,4]
data_SF = pd.merge(table_drp, data_SF)

# PIPE3D
print "Reading Pipe3D Table..."
Table_pip3D = Table.read('manga.Pipe3D-v2_1_2.fits')
table_pip3D = Table_pip3D.to_pandas()
data_SF = pd.merge(table_pip3D,data_SF) 

maps = dict({'127':76,'91':64,'61':54,'37':42,'19':34})
data_SF["fov_pix"] = [maps.get(ifu.astype("str")) for ifu in data_SF["ifudesignsize"]]
data_SF["fov_re"] = data_SF.fov_pix/2.*0.5/data_SF.re_arc

# Classification from vote for Mass Growth CDF
#table_vote = ascii.read('/home/qliu/MaNGA/beta_vote_SFG.dat').to_pandas()
#data_SF = pd.merge(table_vote, data_SF)

# Use Flag from image (Remove merger, center-shifted)
table_use = ascii.read('beta_use_image.dat').to_pandas()
data_SF = pd.merge(table_use[table_use.flag==1],data_SF)

#--------Merging Measurements Over--------#

# Stack G-by-G SGMS
#P1, P2 = data_SF[(data_SF.mode_flag==1)&(data_SF.D4000_slope<-0.0748)], data_SF[(data_SF.mode_flag==-1)&(data_SF.D4000_slope>0.0458)]
P1, P2 = data_SF[data_SF.mode_flag==1], data_SF[data_SF.mode_flag==-1]
P1 = P1[P1.re_arc>2.5]
P2 = P2[P2.re_arc>2.5]
d1_Ms, d1_SFR = pd.Series([]), pd.Series([])
d2_Ms, d2_SFR = pd.Series([]), pd.Series([])

# Shifting
for i,W in enumerate([P1,P2]):
	Plateifu = np.unique(W.plateifu)
	d_Ms, d_SFR = pd.Series([]), pd.Series([])
	for ID in Plateifu[:]:
		G = W[W.plateifu==ID]
		ms_G_med = G.Sigma_Mass.median()
		sfr_G_med = G.Sigma_SFR.median()
		d_Ms = d_Ms.append(G.Sigma_Mass-ms_G_med)
		d_SFR = d_SFR.append(G.Sigma_SFR-sfr_G_med)
	if i==0: d1_Ms = d_Ms;d1_SFR = d_SFR 
	else: d2_Ms = d_Ms;d2_SFR = d_SFR


# Plot Multipanel
m1, sfr1, cof1 = mode_fitting(d1_Ms, d1_SFR, q=0.01, d=6)
m2, sfr2, cof2 = mode_fitting(d2_Ms, d2_SFR, q=0.01, d=6)
# Plotting
fig = plt.figure(figsize=(13,5))
gs1 = mpl.gridspec.GridSpec(1, 5, height_ratios=[1],width_ratios=[17,5,1,17,5])
gs2 = mpl.gridspec.GridSpec(3, 5, height_ratios=[4,5,4],width_ratios=[17,5,1,17,5])
for i,(c,cmap,d_Ms,d_SFR,W,lab) in enumerate(zip(['firebrick','mediumblue'],['Reds','Blues'],[d1_Ms, d2_Ms], [d1_SFR, d2_SFR],[P1,P2],['Inside-out','Outside-in'])):
	#ax = plt.subplot(1,4,i*2+1)
	ax = plt.subplot(gs1[i*3])
	H, xbins, ybins = np.histogram2d(d_Ms, d_SFR,
			bins=(np.linspace(-2.5, 2.25, 50), np.linspace(-2.5, 2.25, 50)))
	#ax.imshow(np.log10(H).T, cmap=cmap, origin='lower',
	 # 		extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]],
	  #		interpolation='nearest', aspect='auto',alpha=0.9)

	H, xbins, ybins = np.histogram2d(d_Ms, d_SFR,
			bins=(np.linspace(-2.5, 2.25, 50), np.linspace(-2.5, 2.25, 50)))

	XH = np.sort(pd.Series(H[H!=0].ravel()))
	Hsum = XH.sum()
	XH_levels = [np.argmin(abs(np.cumsum(XH)-q*Hsum)) for q in [0.01,0.05,0.32,0.7]]
	levels = [XH[k] for k in XH_levels]

	plt.contour(gaussian_filter(H, sigma=.6, order=0).T, levels, extent=[xbins.min(),xbins.max(),
	    ybins.min(),ybins.max()],norm=LogNorm(1, XH_levels[-1]), linewidths=2,cmap=cmap,linestyles='solid')

	pw_bounds = ((d_Ms.quantile(0.3),d_SFR.quantile(0.3),-1.,-1.), 
			 (d_Ms.quantile(0.7),d_SFR.quantile(0.7),3.,3.))

	pw,ew = optimize.curve_fit(piecewise_linear, d_Ms.values, d_SFR.values, 
			p0=[np.median(d_Ms),np.median(d_SFR),1.,1.], bounds=pw_bounds)
	pl,el = optimize.curve_fit(linear, d_Ms.values, d_SFR.values, 
			p0=[1.,0.], bounds=((-1.,-2.),(3.,2)))
	if i==1:
		pw,ew = optimize.curve_fit(piecewise_linear, m2, sfr2, p0=[np.median(m1),np.median(sfr1),1.,1.], bounds=pw_bounds)
		pl,el = optimize.curve_fit(linear, m2, sfr2, 
			p0=[1.,0.], bounds=((-1.,-2.),(3.,2)))
	else:
		pw,ew = optimize.curve_fit(piecewise_linear, m1, sfr1, p0=[np.median(m1),np.median(sfr1),1.,1.], bounds=pw_bounds)
		pl,el = optimize.curve_fit(linear, m1, sfr1, 
			p0=[1.,0.], bounds=((-1.,-2.),(3.,2)))


	#plt.plot(xp, linear(xp, *pl),c='k',ls='-',lw=3,alpha=0.9)	
	plt.plot(xp, piecewise_linear(xp, *pw),c=c,ls='-',lw=4.5,alpha=.9)
	plt.plot(xp, piecewise_linear_b(xp, *pw),c=c,ls='--',lw=3.5,alpha=0.8)
	if i==1:
		plot(m1,sfr1,c='gold',ls='-',marker='o',ms=8, lw=2,alpha=1.)
		plot(m2,sfr2,c='lightgreen',ls='-',marker='o',ms=8, lw=2,alpha=1.)
		text(-1.2,-2.25,"$k_1$:%.2f"%(pw[2]),fontsize=12)
		text(1.48,0.85,"$k_2$:%.2f"%(pw[3]),fontsize=12)
		p=patches.FancyArrowPatch((-1.85, -1.8),(-1.55, -2.2),connectionstyle='arc3, rad=0.5',fc='k',mutation_scale=20,zorder=3);ax.add_patch(p)

		plot(0.,0.,c='deepskyblue',ms=14,marker='*',ls="None",mec='k',mew=.5)
	else:
		plot(m2,sfr2,c='lightgreen',ls='-',marker='o',ms=8, lw=2,alpha=1.)
		plot(m1,sfr1,c='gold',ls='-',marker='o',ms=8, lw=2,alpha=1.)
		text(-2.2,-2.25,"$k_1$:%.2f"%(pw[2]),fontsize=12)
		text(1.4,0.125,"$k_2$:%.2f"%(pw[3]),fontsize=12)
		p=patches.FancyArrowPatch((1.65, 1.55),(2.0, 0.85),connectionstyle='arc3, rad=-0.3',fc='k',mutation_scale=20);ax.add_patch(p)
		
		plot(0.,0.,c='r',ms=14,marker='*',ls="None",mec='k',mew=.5)

	# Chi2/dof
	chi_l=sum(((d_SFR-linear(d_Ms.values,*pl))/ W.e_Sigma_SFR)**2)/(len(W)-2)
	chi_w=sum(((d_SFR-piecewise_linear(d_Ms.values,*pw))/ W.e_Sigma_SFR)**2)/(len(W)-4)

	print 'Chi2/dof 1-component:',chi_l,'Chi2/dof 2-component:',chi_w	
	
	# Std dev
	stdl = np.sqrt(np.sum((linear(d_Ms, *pl)-d_SFR)**2) / d_Ms.size)
	stdw = np.sqrt(np.sum((piecewise_linear(d_Ms.values, *pw)-d_SFR)**2) / d_Ms.size)
	print stdl,stdw

	text(0.05,0.92,lab,fontsize=16,color=c,family='monospace',transform=ax.transAxes)
	xlim(-2.5, 2.25);plt.ylim(-2.5, 2.25)
	xlabel(r'$\rm{\widetilde{\Sigma}_{*} (dex)}$',fontsize='large')
	ylabel(r'$\rm{\widetilde{\Sigma}_{SFR} (dex)}$',fontsize='large')

#IDs=['9026-3703','10001-3702']
IDs=['9041-6102','8611-3702']

for i,(W,ID,xloc,c,cs) in enumerate(zip([P1,P2],IDs,[0.355,0.85],['firebrick','steelblue'],['r','deepskyblue'])):
	#ax = plt.subplot(3,6,3+i*3)
	ax = plt.subplot(gs2[i*3+1])
	img = mpimg.imread('images/%s.png'%ID)
	ax.imshow(img)
	ax.set_title("%s"%ID,fontsize=10,y=-.25)
	axis('off')

for i,(W,ID,xloc,c,cs,clim) in enumerate(zip([P1,P2],IDs,[0.355,0.85],['firebrick','steelblue'],['r','deepskyblue'],[175,31])):
	#ax = plt.subplot(3,6,9+i*3)
	ax = plt.subplot(gs2[i*3+6])
	hdu = fits.open("/home/qliu/MaNGA/SFG_pipe3d/manga-%s.Pipe3D.cube.fits.gz"%ID)
	# Read emission maps
	flux_el = hdu[3].data		
	Ha_map = flux_el[45]
	EWHa_map = flux_el[216]
	#image = abs(EWHa_map[::-1])
	#norm = LogNorm(image.mean() + 0.5 * image.std(), image.max(), clip='True')
	im = imshow(abs(EWHa_map[::-1]),cmap='hot')
	axis('off')
	if i==1:
		coord = [[1,21.5], [10,3], [33,3], [42,21.5], [33,40],[10,40],[1,21.5]]
		xs, ys = zip(*coord); plt.plot(xs,ys,c='m') 
	else:
		coord = [[1.,26.5], [13,4], [40,4], [52.,26.5], [40,49],[13,49],[1,26.5]]
		xs, ys = zip(*coord); plt.plot(xs,ys,c='m') 

	cb=colorbar(im,orientation='horizontal',fraction=0.1,pad=0.,)
	print cb.get_clim()
	cb.set_ticks([0,clim])
	cb.ax.set_title(r"$EW(H\alpha)[\AA]$",fontsize=8,y=-4.5)
	cb.ax.tick_params(labelsize=8)


for i,(W,ID,xloc,c,cs) in enumerate(zip([P1,P2],IDs,[0.355,0.85],['firebrick','steelblue'],['r','deepskyblue'])):
	#ax = plt.subplot(3,6,15+i*3)
	ax = plt.subplot(gs2[i*3+11])
	G = W[W.plateifu==ID]
	ms_G_med = G.Sigma_Mass.median(); sfr_G_med = G.Sigma_SFR.median()
	d_Ms = (G.Sigma_Mass-ms_G_med); d_SFR = (G.Sigma_SFR-sfr_G_med)

	filtered_data, or_fitted_model = or_fit(model_ini, d_Ms, d_SFR)
	p_or = or_fitted_model.parameters
	plot(d_Ms,d_SFR,ms=2.4,marker='o',ls="None",mec=c,mfc='None',mew=0.5,alpha=0.8)
	plot(d_Ms,filtered_data,ms=2,marker='o',ls="None",mec="None", color=c,alpha=0.8)

	axhline(y=0.,xmin=0.,xmax=0.5,color=cs,ls=':',alpha=0.7)
	axvline(x=0.,ymin=0.,ymax=0.5,color=cs,ls=':',alpha=0.7)
	plot(0.,0.,c=cs,ms=7,marker='*',ls="None",mec='k',mew=.5)

	#text(0.08,0.85,ID,fontsize=9,transform=ax.transAxes)
	ax.set_xticks([-2,-1.,0,1.,2.])
	ax.tick_params(axis='both', which='both', labelsize=7,length=2.5, width=1.)
	xlim(-2.5,2.25);ylim(-2.5,2.25)

subplots_adjust(left=0.05, bottom=0.12, right=0.985, top=0.975, wspace=0.15, hspace=0.15)


