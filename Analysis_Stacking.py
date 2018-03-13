import numpy as np
from astropy.io import fits, ascii
from astropy.table import Table, Column
from scipy import optimize, stats
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.patches as patches
import pandas as pd
import seaborn as sns
from astropy.stats import sigma_clip
from astropy.modeling import models, fitting

model_ini = models.Linear1D(1.,0.)
fit = fitting.LevMarLSQFitter()
or_fit = fitting.FittingWithOutlierRemoval(fit, sigma_clip, niter=3, sigma=3.0)

xp = np.linspace(-2.5,2.25,100)

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
table = ascii.read('delta_val_kew.dat')
data = table.to_pandas()
data_SF = data[(data.bpt!=0)&(data.ew_ha_mean<-6)]

# Classification from vote for Mass Growth CDF
#table_vote = ascii.read('beta_vote_SFG.dat').to_pandas()
#data_SF = pd.merge(table_vote, data_SF)

# Use Flag from image (Remove merger, center-shifted)
table_use = ascii.read('beta_use_image.dat').to_pandas()
data_SF = pd.merge(table_use[table_use.flag==1],data_SF)

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

data_SF = pd.merge(table_pip3D,data_SF) 

maps = dict({'127':76,'91':64,'61':54,'37':42,'19':34})
data_SF["fov_pix"] = [maps.get(str(ifu)) for ifu in data_SF["ifudesignsize"]]
data_SF["fov_re"] = data_SF.fov_pix/2.*0.5/data_SF.re_arc

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

#d2_Msb,d2_SFRb,P2b =d2_Ms,d2_SFR,P2 


#m1, sfr1, cof1 = mode_fitting(d1_Ms, d1_SFR, q=0.01, d=6)
#m2, sfr2, cof2 = mode_fitting(d2_Ms, d2_SFR, q=0.01, d=6)

# Contours
fig = plt.figure(figsize=(5,5));
for i,(c,cmap,d_Ms,d_SFR,W,lab) in \
    enumerate(zip(['firebrick','mediumblue'],['Reds','Blues'],
                  [d1_Ms, d2_Ms], [d1_SFR, d2_SFR],[P1,P2],
                  ['Inside-out','Outside-in'])):
	ax = plt.subplot(1,1,1)


	H, xbins, ybins = np.histogram2d(d_Ms, d_SFR,
				bins=(np.linspace(-2.5, 2., 50), np.linspace(-2.5, 2., 50)))

	XH = np.sort(pd.Series(H[H!=0].ravel()))
	Hsum = XH.sum()
	XH_levels = [np.argmin(abs(np.cumsum(XH)-q*Hsum)) for q in [0.01,0.05,0.32,0.7]]
	levels = [XH[k] for k in XH_levels]

	Z = gaussian_filter(H, sigma=.6, order=0).T

	plt.contour(Z, levels, extent=[xbins.min(),xbins.max(), ybins.min(),ybins.max()], norm=LogNorm(1, XH_levels[-1]), linewidths=2,cmap=cmap,linestyles='solid',alpha=0.8,zorder=2)

	text(0.05,0.92-i*0.1,lab,fontsize=16,color=c,family='monospace',transform=ax.transAxes)
	xlim(-2.5, 2.25);plt.ylim(-2.5, 2.25)
	xlabel(r'$\rm{\widetilde{\Sigma}_{*} (dex)}$',fontsize='large')
	ylabel(r'$\rm{\widetilde{\Sigma}_{SFR} (dex)}$',fontsize='large')
tight_layout()

# Contours PSF
fig = plt.figure(figsize=(10,5));
for i,(c,cmap,d_Ms,d_SFR,W,lab) in \
    enumerate(zip(['firebrick','g'],['Reds','Greens'],
                  [d1_Ms0, d1_Ms], [d1_SFR0, d1_SFR],[P10,P1],
                  ['Inside-out FoV>2.5"','Inside-out FoV>5"'])):
	ax = plt.subplot(1,2,1)


	H, xbins, ybins = np.histogram2d(d_Ms, d_SFR,
				bins=(np.linspace(-2.5, 2., 50), np.linspace(-2.5, 2., 50)))

	XH = np.sort(pd.Series(H[H!=0].ravel()))
	Hsum = XH.sum()
	XH_levels = [np.argmin(abs(np.cumsum(XH)-q*Hsum)) for q in [0.01,0.05,0.32,0.7]]
	levels = [XH[k] for k in XH_levels]

	Z = gaussian_filter(H, sigma=.6, order=0).T

	plt.contour(Z, levels, extent=[xbins.min(),xbins.max(), ybins.min(),ybins.max()], norm=LogNorm(1, XH_levels[-1]), linewidths=2,cmap=cmap,linestyles='solid',alpha=0.8,zorder=2)

	text(0.05,0.92-i*0.07,lab,fontsize=16,color=c,family='monospace',transform=ax.transAxes)
	xlim(-2.5, 2.25);plt.ylim(-2.5, 2.25)
	xlabel(r'$\rm{\widetilde{\Sigma}_{*} (dex)}$',fontsize='large')
	ylabel(r'$\rm{\widetilde{\Sigma}_{SFR} (dex)}$',fontsize='large')
for i,(c,cmap,d_Ms,d_SFR,W,lab) in \
    enumerate(zip(['orange','mediumblue'],['Oranges','Blues'],
                  [d2_Ms, d2_Ms0], [d2_SFR, d2_SFR0],[P2,P20],
                  ['Outside-in FoV>5"','Outside-in FoV>2.5"'])):
	ax = plt.subplot(1,2,2)


	H, xbins, ybins = np.histogram2d(d_Ms, d_SFR,
				bins=(np.linspace(-2.5, 2., 50), np.linspace(-2.5, 2., 50)))

	XH = np.sort(pd.Series(H[H!=0].ravel()))
	Hsum = XH.sum()
	XH_levels = [np.argmin(abs(np.cumsum(XH)-q*Hsum)) for q in [0.01,0.05,0.32,0.7]]
	levels = [XH[k] for k in XH_levels]

	Z = gaussian_filter(H, sigma=.6, order=0).T

	plt.contour(Z, levels, extent=[xbins.min(),xbins.max(), ybins.min(),ybins.max()], norm=LogNorm(1, XH_levels[-1]), linewidths=2,cmap=cmap,linestyles='solid',alpha=0.8,zorder=2)

	text(0.05,0.92-i*0.07,lab,fontsize=16,color=c,family='monospace',transform=ax.transAxes)
	xlim(-2.5, 2.25);plt.ylim(-2.5, 2.25)
	xlabel(r'$\rm{\widetilde{\Sigma}_{*} (dex)}$',fontsize='large')
	ylabel(r'$\rm{\widetilde{\Sigma}_{SFR} (dex)}$',fontsize='large')
tight_layout()

# Contours compare
m1, sfr1, cof1 = mode_fitting(d1_Ms, d1_SFR, q=0.01, d=6)
m2, sfr2, cof2 = mode_fitting(d2_Ms, d2_SFR, q=0.01, d=6)
fig = plt.figure(figsize=(5,5));
for i,(c,cmap,d_Ms,d_SFR,W,lab) in \
    enumerate(zip(['firebrick','mediumblue'],['Reds','Blues'],
                  [d1_Ms, d2_Ms], [d1_SFR, d2_SFR],[P1,P2],
                  ['Inside-out','Outside-in'])):
    ax = plt.subplot(1,1,1)

    H, xbins, ybins = np.histogram2d(d_Ms, d_SFR,
    				bins=(np.linspace(-2.5, 2., 50), np.linspace(-2.5, 2., 50)))
    
    XH = np.sort(pd.Series(H[H!=0].ravel()))
    Hsum = XH.sum()
    XH_levels = [np.argmin(abs(np.cumsum(XH)-q*Hsum)) for q in [0.01,0.05,0.32,0.7]]
    levels = [XH[k] for k in XH_levels]
    
    Z = gaussian_filter(H, sigma=.6, order=0).T
    
    plt.contour(Z, levels, extent=[xbins.min(),xbins.max(), ybins.min(),ybins.max()], norm=LogNorm(1, XH_levels[-1]), linewidths=2,cmap=cmap,linestyles='solid',alpha=0.8,zorder=2)
    text(0.05,0.92-i*0.07,lab,fontsize=16,color=c,family='monospace',transform=ax.transAxes)
    xlim(-2.5, 2.25);plt.ylim(-2.5, 2.25)
    xlabel(r'$\rm{\widetilde{\Sigma}_{*} (dex)}$',fontsize='large')
    ylabel(r'$\rm{\widetilde{\Sigma}_{SFR} (dex)}$',fontsize='large')
    pw_bounds = ((d_Ms.quantile(0.3),d_SFR.quantile(0.3),-1.,-1.),
                (d_Ms.quantile(0.7),d_SFR.quantile(0.7),3.,3.))
    pw,ew = optimize.curve_fit(piecewise_linear, d_Ms.values, d_SFR.values, 
		p0=[np.median(d_Ms),np.median(d_SFR),1.,1.], bounds=pw_bounds)
    pl,el = optimize.curve_fit(linear, d_Ms.values, d_SFR.values, 
		p0=[1.,0.], bounds=((-1.,-2.),(3.,2)))
    plt.plot(xp, piecewise_linear(xp, *pw),c=c,ls='-',lw=4.5,alpha=.5)
    #plt.plot(xp, piecewise_linear_b(xp, *pw),c=c,ls='--',lw=3.5,alpha=0.8)
    
    if i==1:
        pw,ew = optimize.curve_fit(piecewise_linear, m2, sfr2, p0=[np.median(m1),np.median(sfr1),1.,1.], bounds=pw_bounds)
        pl,el = optimize.curve_fit(linear, m2, sfr2, 
                p0=[1.,0.], bounds=((-1.,-2.),(3.,2)))
    else:
        pw,ew = optimize.curve_fit(piecewise_linear, m1, sfr1, p0=[np.median(m1),np.median(sfr1),1.,1.], bounds=pw_bounds)
        pl,el = optimize.curve_fit(linear, m1, sfr1, 
                p0=[1.,0.], bounds=((-1.,-2.),(3.,2)))
    if i==1:
        plot(m1,sfr1,c='gold',ls='-',marker='o',ms=8, lw=2,alpha=1.)
        plot(m2,sfr2,c='lightgreen',ls='-',marker='o',ms=8, lw=2,alpha=1.)
        text(-1.2,-2.25,"$k_1$:%.2f"%(pw[2]),fontsize=12)
        text(1.48,0.85,"$k_2$:%.2f"%(pw[3]),fontsize=12)
    else:
        plot(m2,sfr2,c='lightgreen',ls='-',marker='o',ms=8, lw=2,alpha=1.)
        plot(m1,sfr1,c='gold',ls='-',marker='o',ms=8, lw=2,alpha=1.)
        text(-2.2,-2.25,"$k_1$:%.2f"%(pw[2]),fontsize=12)
        text(1.4,0.125,"$k_2$:%.2f"%(pw[3]),fontsize=12)   
tight_layout()


# Shifted SFR/SSFR-Ms
for i,(W,c) in enumerate(zip([P1,P2],['firebrick','steelblue'])):
	Plateifu = np.unique(W.plateifu)
	for ID in Plateifu[:]:
		G = W[W.plateifu==ID]
		ms_G_med = G.Sigma_Mass.median()
		sfr_G_med = G.Sigma_SFR.median()
		d_Ms, d_SFR = pd.Series([]), pd.Series([])
		d_Ms = d_Ms.append(G.Sigma_Mass-ms_G_med)
		d_SFR = d_SFR.append(G.Sigma_SFR-sfr_G_med)
		figure(figsize=(9,4))
		ax=subplot(121);scatter(d_Ms,d_SFR,alpha=0.7,c=c,s=20)
		xlim(-2,1.5);ylim(-2,1.5)
		xlabel(r'$rm{\Delta \Sigma_{*} (dex)}$',fontsize='large')
		ylabel(r'$\rm{\Delta \Sigma_{SFR} (dex)}$',fontsize='large')
		ax=subplot(122);scatter(d_Ms,d_SFR-d_Ms,alpha=0.7,c=c,s=20)
		xlim(-2,1);ylim(-2,1.5)
		xlabel(r'$\rm{\Delta \Sigma_{*} (dex)}$',fontsize='large')
		ylabel(r'$\rm{\Delta \Sigma_{SSFR} (dex)}$',fontsize='large')
		savefig('/home/qliu/MaNGA/beta_G-by-G/%s.png'%ID)
		close()
        

# Cross Validation
from sklearn.model_selection import train_test_split

# 1-Fold CV
df = pd.DataFrame({'m':d2_Ms,'sfr':d2_SFR})
T_set, C_set = train_test_split(df, test_size=0.3)
pt1,et1 = optimize.curve_fit(linear, T_set.m.values, T_set.sfr.values, 
			p0=[1.,0.], bounds=((-1.,-2.),(3.,2)))

pt2,et2 = optimize.curve_fit(piecewise_linear, T_set.m.values, T_set.sfr.values, 
			p0=[np.median(T_set.m),np.median(T_set.sfr),1.,1.], bounds=pw_bounds)

std1 = np.sum((linear(C_set.m, *pt1)-C_set.sfr)**2) / len(C_set)
std2 = np.sum((piecewise_linear(C_set.m.values, *pt2)-C_set.sfr)**2) / len(C_set)
print 'CV error:  1-component:%f,  2-component:%f'%(std1,std2)


# 5-Fold CV
Stds_1, Stds_2 = np.zeros(5), np.zeros(5)
df = pd.DataFrame({'m':d2_Ms,'sfr':d2_SFR})
split_ind = np.linspace(0,len(df),6,dtype=int)
Subsets = np.split(df.sample(frac=1), split_ind[1:-1])

for i in range(5):
	T_set = pd.concat(Subsets[:i] + Subsets[(i+1):])
	C_set = Subsets[i]

	pt1,et1 = optimize.curve_fit(linear, T_set.m.values, T_set.sfr.values, 
			p0=[1.,0.], bounds=((-1.,-2.),(3.,2)))

	pt2,et2 = optimize.curve_fit(piecewise_linear, T_set.m.values, T_set.sfr.values, 
			p0=[np.median(T_set.m),np.median(T_set.sfr),1.,1.], bounds=pw_bounds)

	std1 = np.sum((linear(C_set.m, *pt1)-C_set.sfr)**2) / len(C_set)
	std2 = np.sum((piecewise_linear(C_set.m.values, *pt2)-C_set.sfr)**2) / len(C_set)
	
	Stds_1[i], Stds_2[i] = std1,std2

print '5-Fold CV error:  1-component:%f,  2-component:%f'%(np.median(Stds_1), np.median(Stds_2))

# Learning Rate
N_point = np.arange(100,15000,500)
Stds_1, Stds_2 = np.zeros_like(N_point,dtype=float), np.zeros_like(N_point,dtype=float)
Stds_1t, Stds_2t = np.zeros_like(N_point,dtype=float), np.zeros_like(N_point,dtype=float)
df = pd.DataFrame({'m':d2_Ms,'sfr':d2_SFR})
for i, n in enumerate(N_point):
	T_set, C_set = train_test_split(df.sample(n), test_size=0.3)
	pt1,et1 = optimize.curve_fit(linear, T_set.m.values, T_set.sfr.values, 
			p0=[1.,0.], bounds=((-1.,-2.),(3.,2)))

	pt2,et2 = optimize.curve_fit(piecewise_linear, T_set.m.values, T_set.sfr.values, 
			p0=[np.median(T_set.m),np.median(T_set.sfr),1.,1.], bounds=pw_bounds)

	std1 = np.sum((linear(C_set.m, *pt1)-C_set.sfr)**2) / len(C_set)
	std2 = np.sum((piecewise_linear(C_set.m.values, *pt2)-C_set.sfr)**2) / len(C_set)
	std1t = np.sum((linear(T_set.m, *pt1)-T_set.sfr)**2) / len(T_set)
	std2t = np.sum((piecewise_linear(T_set.m.values, *pt2)-T_set.sfr)**2) / len(T_set)
	print 'CV error:  1-component:%f,  2-component:%f'%(std1,std2)
	Stds_1[i], Stds_2[i] = std1, std2
	Stds_1t[i], Stds_2t[i] = std1t, std2t

figure()
plt.plot(N_point, Stds_1t, c='g',ls ='-')
plt.plot(N_point, Stds_2t, c='y',ls ='-')
plt.plot(N_point, Stds_1, c='g',ls ='--')
plt.plot(N_point, Stds_2, c='y',ls ='--')




