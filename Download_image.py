import pandas as pd
from astropy.io import ascii
from astropy.table import Table
import re
import os

# Read and merge tables
Table_pip3D = Table.read('/home/qliu/MaNGA/pipe3d/manga.Pipe3D-v2_1_2.fits')
table_pip3D = Table_pip3D.to_pandas()
Mass,SFR = table_pip3D.log_mass, table_pip3D.log_sfr_ha

table_vote = ascii.read('/home/qliu/MaNGA/vote.dat')
data = table_vote.to_pandas()
data_vote = pd.merge(table_pip3D[table_pip3D.ew_ha_cen<-3],data)

Differ = data_vote.weight_difference
Ratio = data_vote.weight_ratio
Mode = data_vote.Mode

table_io = data_vote[Mode==1]
table_oi = data_vote[Mode==-1]

# Download images
for ID in table_oi.plateifu:
	plate,ifu = re.findall(r'\d+?\d*e?\d*?',ID)
	print plate, ifu
	os.system("wget -cr 'https://data.sdss.org/sas/dr14/manga/spectro/redux/v2_1_2/%s/stack/images/%s.png' -O %s.png"%(plate,ifu,ID))    

for ID in table_io.plateifu:
	plate,ifu = re.findall(r'\d+?\d*e?\d*?',ID)
	print plate, ifu
	os.system("wget -cr 'https://data.sdss.org/sas/dr14/manga/spectro/redux/v2_1_2/%s/stack/images/%s.png' -O %s.png"%(plate,ifu,ID)) 

                                    
