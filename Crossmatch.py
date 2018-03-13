import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
import pandas as pd
from astropy.table import Table
from astroML.crossmatch import crossmatch_angular
from astroML.plotting import hist

# Read and merge MPA-JHU DR8 catalogs
print "Reading catalogs..."
Table_info = Table.read('/home/qliu/MaNGA/galSpecInfo-dr8.fits')
table_info = Table(Table_info.columns["SPECOBJID","RA","DEC"]).to_pandas()

Table_ex = Table.read('/home/qliu/MaNGA/galSpecExtra-dr8.fits')
table_ex = Table(Table_ex.columns["SPECOBJID","BPTCLASS",'LGM_TOT_P50','SFR_TOT_P50']).to_pandas()

SF_cond = (table_ex.BPTCLASS==1)|(table_ex.BPTCLASS==2)|(table_ex.BPTCLASS==3)

table_SFG_sdss = table_info[SF_cond] 
table_SFG_sdss["bptclass"] = table_ex.BPTCLASS[SF_cond]
table_SFG_sdss["lgm_tot"] = table_ex.LGM_TOT_P50[SF_cond]
table_SFG_sdss["lgsfr_tot"] = table_ex.SFR_TOT_P50[SF_cond]

# Set arrays of MPA-JHU SFG
SDSSX = np.empty((len(table_SFG_sdss), 2), dtype=np.float64)
SDSSX[:, 0] = table_SFG_sdss['RA']
SDSSX[:, 1] = table_SFG_sdss['DEC']

# Set arrays of MaNGA DR14
Table_drp = Table.read('/home/qliu/MaNGA/drpall-v2_1_2.fits')
table_drp = Table(Table_drp.columns["plateifu","objra","objdec"]).to_pandas()
MANGAX = np.empty((len(table_drp), 2), dtype=np.float64)
MANGAX[:, 0] = table_drp['objra']
MANGAX[:, 1] = table_drp['objdec']

# Crossmatch catalogs
print "Start Crossmatching..."
max_radius = 1. / 3600  # 1 arcsec
dist, ind = crossmatch_angular(SDSSX, MANGAX, max_radius)
match = ~np.isinf(dist)

dist_match = dist[match]
dist_match *= 3600

# Plot crossmatching results
ax = plt.axes()
hist(dist_match, bins='knuth', ax=ax,
     histtype='stepfilled', ec='k', fc='#AAAAAA')
ax.set_xlabel('radius of match (arcsec)')
ax.set_ylabel('N(r, r+dr)')
ax.text(0.95, 0.95,"MPA-JHU DR8 objects: %i" % SDSSX.shape[0],
        ha='right', va='top', transform=ax.transAxes)
ax.text(0.95, 0.9,"MaNGA DR14 objects: %i" % MANGAX.shape[0],
        ha='right', va='top', transform=ax.transAxes)
ax.text(0.95, 0.85,"Number with match: %i" % np.sum(match),
        ha='right', va='top', transform=ax.transAxes)
plt.title("Crossmatching SFG MaNGA DR14 / MPA-JHU DR8")

# merge table by ra,dec
def Cord_to_Plateifu(ra,dec):
    ii = np.where( (abs(table_drp['objra']-ra) < 0.001) & (abs(table_drp['objdec']-dec) < 0.001) )[0]
    return table_drp.iloc[ii].plateifu

# Save records
table_match = table_SFG_sdss[match] 

Plateifu = [Cord_to_Plateifu(ra,dec).values[0] for (ra,dec) in zip(table_match.RA.values, table_match.DEC.values)]

table_match["plateifu"] = Plateifu

save = True
if save:
	table_match.to_csv("crossmatch_drp.dat", sep= " ", index=False)
