import numpy as np
import pandas as pd

#%%
def LatticeSiteLabeling(unitcell):
    '''Applies a logical framework to determine projected lattice site type
    for each row of a Pandas DataFrame that contains atomic column coordinates
    and metadata. A column is added to the DataFrame to hold the lattice site
    information. This framework must be modified for each new structure and 
    orientation. Additionally, it can be programed to return a dictionary of 
    rich text strings (PltLabels) to use as labels when plotting.'''
    lattice_site=[]

    for ind in unitcell.index:
        if unitcell.at[ind,'elem'] == 'Sm/Ba':
            lattice_site.append('A1')
        elif unitcell.at[ind,'elem'] == 'Ba':
            if unitcell.at[ind,'u'] == 0.5:
                lattice_site.append('A2_1')
            else:
                lattice_site.append('A2_2')
        elif unitcell.at[ind,'elem'] == 'Sn/Nb':
            if unitcell.at[ind,'u'] == 0.5:
                lattice_site.append('B2')
            else:
                lattice_site.append('B1')
        elif unitcell.at[ind,'elem']== 'O':
            lattice_site.append('O')
            
        else: print(unitcell.loc[ind,:])
    
    unitcell.loc[:,'LatticeSite']=lattice_site
    return unitcell

    