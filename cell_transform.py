import numpy as np
import matplotlib.pyplot as plt
import  matplotlib.patches as patches
from CifFile import ReadCif
import pandas as pd
from SingleOrigin.utils import *
import copy

'''*** Requirements for using this script ***
    1) .cif file saved in the cifdir
    2) .txt file with all atom positions generated by "cif2cell" saved in the
     cifdir
    3) Specify basis vectors of a new projected unit cell:
            --ZA is the zone axis direction
            --a2_p is the ~x direction in desired projection
            --a3_p is the ~y direction in desired projection
            **a2_p & a3_p should create a 2D unit cell which encompasses the
                projected extents of the original unit cell, e.g. for a
                [110] ZA in cubic, [-110] and [001] should be choosen for a2_p
                and a3_p.
    4) If you want to use lattice site labels, adjust the "Add site labeling" 
     cell to be appropriate for your structure, otherwise comment-out the
     section.
    5) Adjust directory and file name information in this block'''
                
#Specify cif file and desired zone axis
cif='BSSN5137'

class UnitCell():
    def __init__(self, cif_data, origin_shift = [0, 0, 0]):
        Crystal = cif_data.dictionary[list(cif_data.dictionary.keys()
                                                )[0]].block

        self.a1 = float(Crystal['_cell_length_a'][0].split("(", 1)[0])
        self.a2 = float(Crystal['_cell_length_b'][0].split("(", 1)[0])
        self.a3 = float(Crystal['_cell_length_c'][0].split("(", 1)[0])
        self.alpha = float(Crystal['_cell_angle_alpha'][0].split("(", 1)[0])
        self.beta = float(Crystal['_cell_angle_beta'][0].split("(", 1)[0])
        self.gamma = float(Crystal['_cell_angle_gamma'][0].split("(", 1)[0])
        coord_max_precision = max([
            max([len(xi) for xi in Crystal['_atom_site_fract_x'][0]]),
            max([len(yi) for yi in Crystal['_atom_site_fract_y'][0]]),
            max([len(zi) for zi in Crystal['_atom_site_fract_z'][0]])]) - 2
        
        if coord_max_precision < 2: coord_max_precision=2
            
        xyz = ((np.array([
            [i.split('(', 1)[0] for i in Crystal['_atom_site_fract_x'][0]],
            [i.split('(', 1)[0] for i in Crystal['_atom_site_fract_y'][0]],
            [i.split('(', 1)[0] for i in Crystal['_atom_site_fract_z'][0]]],
            dtype = np.float64) 
            + np.array(origin_shift, ndmin=2).T) % 1).T
        elem = np.array(Crystal['_atom_site_type_symbol'][0],
                        ndmin=2).T
        site_frac = np.array(Crystal['_atom_site_occupancy'][0],
                             ndmin=2, dtype=str).T
        
        atoms=pd.DataFrame({'u': xyz[:,0], 'v': xyz[:,1], 'w': xyz[:,2],
                                 'elem': elem[:,0],
                                 'site_frac': site_frac[:,0]})
        
        # Sort and combine atoms if multiple at one position
        atoms = atoms.sort_values(['u', 'v', 'w'])
        atoms.reset_index(inplace=True, drop=True)
        # self.atoms=atoms
        _, index, rev_index, counts = np.unique(atoms.loc[:, 'u':'w'], 
                                             axis=0,
                                             return_index=True,
                                             return_inverse=True, 
                                             return_counts=True)
        atoms_ = atoms.iloc[index,:].copy()
        for i, ind in enumerate(index):
            if counts[i] > 1:
                for j in range(counts[i]-1):
                    atoms_.loc[ind,'elem'] += ('/' + atoms.at[np.sum(counts[:i])+1+j, 'elem'])
                    atoms_.loc[ind,'site_frac'] += ('/' + atoms.at[np.sum(counts[:i])+1+j, 'site_frac'])
        atoms = atoms_.copy()
        del atoms_
        
        # Generate all atoms in unit cell using symmetry related positions
        for n, atom in atoms.iterrows():
            xyz0 = np.array([atom.u, atom.v, atom.w])
            
            for oper in Crystal['_symmetry_equiv_pos_as_xyz'][0]:
                xyz_ = np.array(2*np.ones(3))
                while np.allclose(xyz0, xyz_, atol=0.1) == False:
                    if xyz_[0] == 2:
                        xyz_ = xyz0.copy()
                    [x,y,z] = xyz_
                    xyz_ = np.array([eval(oper.split(',')[0]), 
                                      eval(oper.split(',')[1]),
                                      eval(oper.split(',')[2])]) % 1
                    new_pos = atom.copy()
                    [new_pos.at['u'], new_pos.at['v'], new_pos.at['w']] = [
                        *xyz_]
                    atoms = atoms.append(new_pos, ignore_index=True)

        print(atoms.shape)

        atoms = atoms.loc[atoms.round({'u': 3, 'v': 3, 'w': 3}
                                      ).drop_duplicates().index, :]
        atoms = atoms.reset_index(drop=True)
        print(atoms.shape)
        
        self.atoms=atoms
        self.a_3d = None
        
        self.at_cols = None
        self.a_2d = None
                        
        #Find metric tensor
        self.g = metric_tensor(self.a1, self.a2, self.a3, self.alpha,
                                   self.beta, self.gamma)
        self.g_star = np.linalg.inv(self.g)

    def transform_basis(self, za, a2_, a3_):
        '''Expand crystal to 5x5x5 original unit cells, change basis to new
        reference frame and reduce to only positions in the new unit cell.'''
        a2_ = a2_ / (np.min(np.abs(a2_)[np.abs(a2_)>0]) * np.sum(np.abs(a2_)))
        a3_ = a3_ / (np.min(np.abs(a3_)[np.abs(a3_)>0]) * np.sum(np.abs(a3_)))
        alpha_t=np.array([za, a2_, a3_])
        g_=alpha_t @ alpha_t.T @ self.g
        g_star_=np.linalg.inv(g_)
        V=np.sqrt(np.linalg.det(g_))
        a_3d=np.array([[np.sqrt(g_[0,0]), g_[1,0]/np.sqrt(g_[0,0]),
                          g_[2,0]/np.sqrt(g_[0,0])],
                         [0, V*np.sqrt(g_star_[2,2]/g_[0,0]),
                          -V*g_star_[2,1]/np.sqrt(g_star_[2,2]*g_[0,0])],
                         [0, 0, 1/np.sqrt(g_star_[2,2])]])
        # latt_cells = np.array([[i, j, k] for i in range(-3, 4)
        #                                  for j in range(-3, 4)
        #                                  for k in range(-3, 4)])
        
        # latt_cells_ = np.array([row for row in latt_cells 
        #                        for i in range(self.atoms.shape[0])])
        
        # atoms = copy.deepcopy(pd.DataFrame(columns = (list(self.atoms))))
        # for i in range(latt_cells.shape[0]):
        #     atoms = atoms.append(self.atoms)
        # atoms.reset_index(drop=True, inplace=True)
        atoms = copy.deepcopy(self.atoms)
        atoms.loc[:, 'u':'w'] = (atoms.loc[:, 'u':'w'].to_numpy()
                                 @ np.linalg.inv(alpha_t)) % 1
        
        '''Transform coordinates to Cartesian using the structure matrix'''
        atoms.insert(atoms.shape[1], 'x', 0)
        atoms.insert(atoms.shape[1], 'y', 0)
        atoms.insert(atoms.shape[1], 'z', 0)
        atoms.loc[:, 'x':'z'] = atoms.loc[:, 'u':'w'].to_numpy() @ a_3d.T
        atoms.reset_index(drop=True, inplace=True)
        self.atoms = atoms
        self.a_3d = a_3d
        self.g = g_
        self.g_star = g_star_
        # return atoms, g_, a_3d
    
    def project_uc_2d(self, proj_axis=0):
        '''Create DataFrame of atom columns from 3D atom positions;
        add 'weight' as number of atoms per column. "proj_axis" must be one of
        the basis vectors of the original or transformed unit cell'''
        
        at_col = copy.deepcopy(self.atoms)
        a_2d=np.delete(self.a_3d, proj_axis, axis=0)
        a_2d=np.delete(a_2d, proj_axis, axis=1)
        cart_ax = ['x', 'y', 'z']
        cryst_ax = ['u', 'v', 'w']
        at_col = at_col.drop([cart_ax[proj_axis], cryst_ax[proj_axis]], axis = 1)
        cryst_ax.pop(proj_axis)
        elem = [''.join(item) for item in at_col.loc[:,'elem']]
        x = list(at_col.loc[:, cryst_ax[0]])
        y = list(at_col.loc[:, cryst_ax[1]])
        (_, index, counts) =np.unique(np.array(list(zip(x, y, elem))),
                                return_index=True, return_counts=True, axis=0)
        weights=pd.DataFrame(list(zip(index,counts))).sort_values(by=[0], 
                                                    ignore_index=True)
        index = weights.iloc[:,0].tolist()
        at_col = at_col.iloc[index, :]
        at_col.loc[:,'weight']=list(weights.loc[:,1])
        
        at_cols=copy.deepcopy(at_col)
        at_cols=at_col.reset_index(drop=True)
        
        cryst_ax = list(set(cryst_ax).intersection(
            set(at_cols.columns.to_list())))
        cryst_ax.sort()
        at_cols.rename(columns={k: v for (k,v) in zip(cryst_ax, ['u', 'v'])}, 
                       inplace=True)
        
        cart_ax = list(set(cart_ax).intersection(
            set(at_cols.columns.to_list())))
        cart_ax.sort()
        at_cols.rename(columns={k: v for (k,v) in zip(cart_ax, ['x', 'y'])}, 
                       inplace=True)
        
        self.at_cols = at_cols
        self.a_2d = a_2d
        # return at_col, a_2d
    
    def combine_prox_cols(self, prox_toler=0.1):
        '''Combine coincident (or proximate) columns in 2D projected unit cell
            even if different atom type'''
        
        '''Loop through all pairwise combinations of rows, find pairs that 
            are close'''
        prox_rows = []
        for i, row_i in self.at_cols.iterrows():
            for j, row_j in self.at_cols.iterrows():
                dist = np.linalg.norm((np.array([row_i.at['x'], 
                                                 row_i.at['y']]) 
                                      - np.array([row_j.at['x'], 
                                                  row_j.at['y']])))
                
                if i == j: break
                
                elif dist < prox_toler:
                    print(dist)
                    prox_rows.append([j,i])
        print(prox_rows)
        '''Combine close-pairs into minimum unique sets. 
            i.e. close pairs [1,2], [2,3], and [3,1], are actually a close 
            triplet, [1,2,3]'''
        test = False
        while test == False:
            combined = []
            for i, rowsi in enumerate(prox_rows):
                for j, rowsj in enumerate(prox_rows):
                    if (set(rowsi)&set(rowsj)):
                        combined.append(list(set(rowsi + rowsj)))
            
            combined = np.array(np.unique(combined, axis=0), ndmin=2).tolist()
            test = (np.unique([y for x in combined for y in x]).shape 
                    == np.array([y for x in combined for y in x]).shape)
            prox_rows = copy.deepcopy(combined)
        print(prox_rows)
        '''Use close-sets to combine dataframe rows and retain all site info'''
        s = '|'
        new = []
        for i, rows in enumerate(prox_rows):
            new.append(self.at_cols.loc[rows[0], :])
            new[i].at['u'] = np.average([self.at_cols.at[ind, 'u'] 
                                         for ind in rows])
            new[i].at['v'] = np.average([self.at_cols.at[ind, 'v'] 
                                         for ind in rows])
            new[i].at['x'] = np.average([self.at_cols.at[ind, 'x'] 
                                         for ind in rows])
            new[i].at['y'] = np.average([self.at_cols.at[ind, 'y'] 
                                         for ind in rows])
            elem = [self.at_cols.at[ind, 'elem'] for ind in rows]
            new[i].at['elem'] = s.join(elem)
            site_frac = [self.at_cols.at[ind, 'site_frac'] for ind in rows]
            new[i].at['site_frac'] = s.join(site_frac)
            self.at_cols = self.at_cols.drop(rows, axis=0)
            
        self.at_cols = self.at_cols.append(new)
        self.at_cols = self.at_cols.reset_index(drop=True)
        return self.at_cols

def Lattice2d_plot(at_col, a_2d, SiteLabels=None,
                   exclude_site=[], proj_axis = 0):
    if SiteLabels: #'LatticeSite' in at_col.columns:
        lab = 'LatticeSite'
        ColList = list(set(at_col[lab]))
        print(ColList)
        ColList.sort()
        
    else:
        lab = 'elem'
        s='/'
        ColList = [s.join(x) for x in list(set(at_col[lab]))]
        ColList.sort()
        
    ColList = [site for site in ColList if site not in exclude_site]
    cmap = plt.get_cmap('RdYlGn')
    if len(ColList) > 1:
        c = {ColList[x] : np.array([cmap(x/(len(ColList)-1))],
                                   ndmin=2) for x in range(len(ColList))}
    else:
        c = {ColList[0] : 'red'}
    fig = plt.figure()
    ax=fig.add_axes([0.1, 0.1, 0.5, 0.75])
    ax.set_aspect(1)
    p = patches.Polygon(np.array([[0, 0],
                                  [a_2d[0,0], 0],
                                  [a_2d[0,0]+a_2d[0,1], a_2d[1,1]],
                                  [a_2d[0,1], a_2d[1,1]]]),
                        fill = False, ec = 'black', lw = 0.5)
    ax.add_patch(p)
    ax.set_facecolor('grey')
    
    cart_ax = list(set(at_col.columns.tolist()) & {'x', 'y', 'z'})
    cart_ax.sort()
    if SiteLabels == None:
        for site in ColList:
            ax.scatter(at_col.loc[:, cart_ax[0]],
                       at_col.loc[:, cart_ax[1]],
                       c=c[site], vmin = 0, vmax = 1, s=2,
                       label = at_col[lab])
        ax.legend(bbox_to_anchor=(1.1, 0.5), loc=6, borderaxespad=0.,
                  facecolor='grey')
        
    else:
        for site in ColList:
            ax.scatter(at_col.loc[at_col[lab] == site].loc[:, cart_ax[0]],
                       at_col.loc[at_col[lab] == site].loc[:, cart_ax[1]],
                       c=c[site], vmin = 0, vmax = 1, s=2,
                       label = SiteLabels[site]['label'])
        ax.legend(bbox_to_anchor=(1.1, 0.5), loc=6, borderaxespad=0.,
                  facecolor='grey')
