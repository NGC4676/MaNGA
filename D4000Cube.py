import numpy as np
import astropy
from astropy.io import fits, ascii
from astropy.table import Table, Column
import matplotlib.pyplot as plt
from scipy import optimize, stats
import pandas as pd
import re

# Read DRP / pipe3D catalog
print "Reading MaNGA DRP Table..."
Table_drp = Table.read('/home/qliu/MaNGA/drpall-v2_1_2.fits')
table_drp = Table(Table_drp.columns["plateifu","nsa_sersic_ba"]).to_pandas()
table_drp['NUV_r'] =Table_drp.columns['nsa_sersic_absmag'][:,1] - Table_drp.columns['nsa_sersic_absmag'][:,4]

print "Reading Pipe3D Table..."
Table_pip3D = Table.read('/home/qliu/MaNGA/pipe3d/manga.Pipe3D-v2_1_2.fits')
table_pip3D = Table_pip3D.to_pandas()

print "Merging catalogs... MaNGA DRP"
table_merg = pd.merge(table_pip3D,table_drp)
table_face = table_merg[table_merg.nsa_sersic_ba>0.5]     # face-on (b/a > 0.5)

print "Merging catalogs... Pipe3D + MPA-JHU SFG"
table_match = ascii.read('/home/qliu/MaNGA/crossmatch_drp.dat').to_pandas()
table_SFG = pd.merge(table_face,table_match).drop_duplicates(subset="plateifu") #remove repeated crossmacthing object

print "SFG selection: NUV-r < 4"
table_SFG = table_SFG[table_SFG.NUV_r<4]

dir=["/home/qliu/MaNGA/SFG_pipe3d/%s.Pipe3D.cube.fits.gz"%ID for ID in table_SFG.mangaid]

f=dir[0]
ID = re.findall(r'\d+?-\d*e?\d*?',f)[0]
hdu = fits.open(f)

# Read ssp maps
ssp = hdu[1].data        
	#SSP[ID] = ssp 
Ms_map = ssp[19]
	#Ms_maps[ID] = Ms_map
Binning_map = ssp[1]
bins = np.unique(Binning_map)

absp = hdu[4].data 
D4000_map = ssp[5]
