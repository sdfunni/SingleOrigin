"""SingleOrigin is a module for atomic column position finding intended for 
    high resolution scanning transmission electron microscope images.
    Copyright (C) 2022  Stephen D. Funni

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see https://www.gnu.org/licenses"""


import warnings
import copy
import numpy as np
import matplotlib.pyplot as plt
import  matplotlib.patches as patches
from CifFile import ReadCif
import pandas as pd
from SingleOrigin.utils import (metric_tensor, bond_length, bond_angle)

#%%
class UnitCell():
    """Object class for defining and manipulating the unit cell from a .cif.
    
    Class with methods for defining a unit cell from a .cif file and 
    manipulating the definition.
    
    Parameters
    ----------
    cif_path : str
        Path to the .cif file.
    origin_shift : array_like with shape (3,)
        Change of origin in fractional coordinates.
        Default: [0, 0, 0].
    
    Attributes
    ----------
    a, b, c, alpha, beta, gamma : The lattice parameters.
    atoms : DataFrame containing all the atoms in the unit cell.
    at_cols : The 2D atom column positions generated by the 'project_uc_2d()'
        method.
    a_3d_real : The 3D transformation matrix from fractional to real 
        coordinates (Angstroms).
    a_2d_real : The 2D transformation matrix from fractional to real 
        coordinates (Angstroms). Generated by the 'project_uc_2d()' method.
    g : The direct metric tensor.
    g_star : The reciprocal metric tensor.
    alpha_t = The transformation matrix from the original basis system
        of the .cif file to the "new" basis specified by the 'transform_basis()'
        method.
    
    Methods
    -------
    transform_basis(za, a1_, a2_) : Change the crystallographic basis.
    project_uc_2d(proj_axis=0, ignore_elements = []) : Project the unit cell
        along a certain basis vector direction.
    combine_prox_cols(toler=0.1) : Combines atom columns if closer than 
        the toler.
    plot_unit_cell(label='elem') : Plots the projected unit cell for 
        verification.
    
    """
    
    def __init__(self, cif_path, origin_shift = [0, 0, 0]):
        cif_data = copy.deepcopy(ReadCif(cif_path))
        Crystal = cif_data.dictionary[list(cif_data.dictionary.keys()
                                                )[0]].block

        self.a = float(Crystal['_cell_length_a'][0].split("(", 1)[0])
        self.b = float(Crystal['_cell_length_b'][0].split("(", 1)[0])
        self.c = float(Crystal['_cell_length_c'][0].split("(", 1)[0])
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
        _, index, rev_index, counts = np.unique(atoms.loc[:, 'u':'w'], 
                                             axis=0,
                                             return_index=True,
                                             return_inverse=True, 
                                             return_counts=True)
        atoms_ = atoms.iloc[index,:].copy()
        for i, ind in enumerate(index):
            if counts[i] > 1:
                for j in range(counts[i]-1):
                    atoms_.loc[ind,'elem'] += (
                        '/' + atoms.at[np.sum(counts[:i])+1+j, 'elem'])
                    atoms_.loc[ind,'site_frac'] += (
                        '/' + atoms.at[np.sum(counts[:i])+1+j, 'site_frac'])
        atoms = atoms_.copy()
        del atoms_
        
        if '_symmetry_equiv_pos_as_xyz' in Crystal.keys():
            symm_eq_pos = Crystal['_symmetry_equiv_pos_as_xyz'][0]
        elif '_space_group_symop_operation_xyz' in Crystal.keys():
           symm_eq_pos = Crystal['_space_group_symop_operation_xyz'][0]
           
        else: 
           symm_eq_pos = []
           warnings.warn('No symmetry related positions found. Check'
                         + 'structure to ensure accuracy.')
        # Generate all atoms in unit cell using symmetry related positions
        if len(symm_eq_pos) > 1:
            for n, atom in atoms.iterrows():
                xyz0 = np.array([atom.u, atom.v, atom.w])
                
                for oper in symm_eq_pos:
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

        atoms = atoms.loc[atoms.round({'u': 3, 'v': 3, 'w': 3}
                                      ).drop_duplicates().index, :]
        atoms = atoms.reset_index(drop=True)
        
        self.atoms=atoms
        self.at_cols = pd.DataFrame
        
        g = metric_tensor(self.a, self.b, self.c, self.alpha,
                                   self.beta, self.gamma)
        g_star = np.linalg.inv(g)
        V = np.sqrt(np.linalg.det(g))
        self.a_3d = np.array([
            [g[0,0]**0.5, g[1,0]/g[0,0]**0.5, g[2,0]/g[0,0]**0.5],
            [0, V*(g_star[2,2]/g[0,0])**0.5, 
             -V*g_star[2,1]/(g_star[2,2]*g[0,0])],
            [0, 0, g_star[2,2]**-0.5]])
        
        atoms.insert(atoms.shape[1], 'x', 0)
        atoms.insert(atoms.shape[1], 'y', 0)
        atoms.insert(atoms.shape[1], 'z', 0)
        atoms.loc[:, 'x':'z'] = atoms.loc[:, 'u':'w'].to_numpy() @ self.a_3d.T
        atoms.reset_index(drop=True, inplace=True)
        
        self.at_cols = None
        self.a_2d = None
        self.g = g
        self.g_star = g_star
        self.alpha_t = np.identity(3)
        

    def transform_basis(self, a1_, a2_, a3_):
        """Change the crystallographic basis.
        
        Modifies the class instance to a new crystallographic basis.
        
        Parameters
        ----------
        za, a1_, a2_: array_like with shapes (3,)
            Miller-Bravis indices of the "new" basis vectors in terms of 
            the "old" basis vectors.
            
        Returns
        -------
        None.
            
        """
        
        alpha_t=np.array([a1_, a2_, a3_])
        
        a1_mag = bond_length([0,0,0], alpha_t[0], self.g)
        a2_mag = bond_length([0,0,0], alpha_t[1], self.g)
        a3_mag = bond_length([0,0,0], alpha_t[2], self.g)
        alpha_ = bond_angle(alpha_t[1], [0,0,0], alpha_t[2], self.g)
        beta_ = bond_angle(alpha_t[0], [0,0,0], alpha_t[2], self.g)
        gamma_ = bond_angle(alpha_t[0], [0,0,0], alpha_t[1], self.g)
        print('transformed lattice parameters:', '\n',
              'a1:    ', np.around(a1_mag, decimals=4), '\n',
              'a2:    ', np.around(a2_mag, decimals=4), '\n',
              'a3:    ', np.around(a3_mag, decimals=4), '\n',
              'alpha: ', np.around(alpha_, decimals=4), '\n',
              'beta:  ', np.around(beta_,  decimals=4), '\n',
              'gamma: ', np.around(gamma_, decimals=4), '\n')
        g = metric_tensor(a1_mag, a2_mag, a3_mag, alpha_, beta_, gamma_)
        g_star =np.linalg.inv(g)
        
        V = np.sqrt(np.linalg.det(g))
        self.a_3d = np.array([
            [g[0,0]**0.5, g[1,0]/g[0,0]**0.5, g[2,0]/g[0,0]**0.5],
            [0, V*(g_star[2,2]/g[0,0])**0.5, 
             -V*g_star[2,1]/(g_star[2,2]*g[0,0])],
            [0, 0, g_star[2,2]**-0.5]])
        
        latt_cells = np.array([[i, j, k] for i in range(-3, 4)
                                         for j in range(-3, 4)
                                         for k in range(-3, 4)])
        
        latt_cells_ = np.array([row for row in latt_cells 
                               for i in range(self.atoms.shape[0])])
        
        atoms = pd.DataFrame(columns = (list(self.atoms)))
        for i in range(latt_cells.shape[0]):
            atoms = atoms.append(self.atoms)
        atoms.reset_index(drop=True, inplace=True)
        atoms.loc[:, 'u':'w'] = (atoms.loc[:, 'u':'w'].to_numpy()
                                 + latt_cells_) @ np.linalg.inv(alpha_t)
        atoms.loc[:, 'u':'w'] = np.around(atoms.loc[:, 'u':'w'], decimals=5)
        
        '''Reduce to new unit cell'''
        atoms = atoms[((atoms.u >= 0) & (atoms.u < 1) &
                       (atoms.v >= 0) & (atoms.v < 1) &
                       (atoms.w >= 0) & (atoms.w < 1))]
        
        '''Transform coordinates to Cartesian using the structure matrix'''
        
        atoms.loc[:, 'x':'z'] = atoms.loc[:, 'u':'w'].to_numpy() @ self.a_3d.T
        atoms.reset_index(drop=True, inplace=True)
        self.atoms = atoms
        self.g = g
        self.g_star = g_star
        self.alpha_t = alpha_t
    
    def project_uc_2d(self, proj_axis=0, ignore_elements = [],
                      unique_proj_cell=True):
        """Project the unit cell along a certain basis vector direction.
        
        Generates a 2D projected unit cell, assigning the result to the 
        class attribute 'at_cols'. The projected basis vector matrix is 
        assigned to 'a_2d'.
        
        Parameters
        ----------
        proj_axis : int
            The basis vector along which to project the unit cell. 0, 1, or 2. 
            Value corresponds to the current basis vector definitions, 
            i.e. a1, a2, and a3. 
            Default: 0.
        ignore_elements : list of strings
            The element labels of the atoms that should be dropped before 
            projecting the unit cell. For example, light elements such as 
            oxygen are not visible in HAADF STEM and can be removed from
            the projection.
            Default = []
        unique_proj_cell : bool
            Whether to reduce projected unit cell to minimum required. For 
            example, [110] zone axis silicon has [-110] axis in plane, but 
            only the projected cell bounded by the 1/2 [-110] & [001] basis   
            vectors is needed as the two halves of the cell are identical once 
            projected. 
            Default: True.
            
        Returns
        -------
        None.
            
        """
        
        for ignore_element in ignore_elements:
            self.atoms = self.atoms[self.atoms.elem != ignore_element]
        
        self.alpha_t = self.alpha_t[[(proj_axis + i) % 3 for i in range(3)], :]
         
        at_col = self.atoms.copy()
        
        a_2d=np.delete(self.a_3d, proj_axis, axis=0)
        a_2d=np.delete(a_2d, proj_axis, axis=1)
        cart_ax = ['x', 'y', 'z']
        cryst_ax = ['u', 'v', 'w']
        at_col = at_col.drop([cart_ax[proj_axis], 
                              cryst_ax[proj_axis]], axis = 1)
        cryst_ax.pop(proj_axis)
        elem = [''.join(item) for item in at_col.loc[:,'elem']]
        x = list(at_col.loc[:, cryst_ax[0]])
        y = list(at_col.loc[:, cryst_ax[1]])
        (_, index, counts) =np.unique(np.array(list(zip(x, y, elem))),
                                return_index=True, return_counts=True, axis=0)
        weights = pd.DataFrame(list(zip(index,counts))).sort_values(by=[0], 
                                                    ignore_index=True)
        index = weights.iloc[:,0].tolist()
        at_col = at_col.iloc[index, :]
        at_col.loc[:,'weight']=list(weights.loc[:,1])
        
        at_cols = at_col.copy()
        at_cols = at_col.reset_index(drop=True)
        
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

        if unique_proj_cell == True:
            a1_ = self.alpha_t[1,:]
            a2_ = self.alpha_t[2,:]
            
            #Find frequency of planes along each basis vector
            a1_mult = (np.min(np.abs(a1_)[np.abs(a1_)>0]) 
                       * np.sum(np.abs(a1_)))
            a2_mult = (np.min(np.abs(a2_)[np.abs(a2_)>0]) 
                       * np.sum(np.abs(a2_)))
            
            #Convert fractional coordinates to new basis magnitudes
            at_cols.loc[:, 'u':'v'] *= np.array([a1_mult, a2_mult])

            at_cols = at_cols[((at_cols.u >= 0) & (at_cols.u < 1) &
                             (at_cols.v >= 0) & (at_cols.v < 1))]
            a_2d *= np.array([[1 / a1_mult, 1 / a2_mult]]).T
            
        at_cols = at_cols.sort_values(by=['elem', 'u', 'v'])
        at_cols.reset_index(drop=True, inplace=True)
        at_cols.reset_index(drop=True, inplace=True)
        at_cols.reset_index(drop=False, inplace=True)
        
        self.at_cols = at_cols
        self.a_2d = a_2d
    
    def combine_prox_cols(self, toler=0.1):
        """Combines atom columns if closer than toler (in Angstroms).
        
        Generates a 2D projected unit cell, assigning the result to the 
        class attribute 'at_cols'. The proejected basis vector matrix is 
        assigned to 'a_2d'.
        
        Parameters
        ----------
        toler : float
            The threshold distance tolerance for combining atom columns.
            Default: 0.1.
            
        Returns
        -------
        None.
            
        """
            
        print('Distances being combined...')
        prox_rows = []
        for i, row_i in self.at_cols.iterrows():
            for j, row_j in self.at_cols.iterrows():
                dist = np.linalg.norm((np.array([row_i.at['x'], 
                                                 row_i.at['y']]) 
                                      - np.array([row_j.at['x'], 
                                                  row_j.at['y']])))
                
                if i == j: break
                
                elif dist < toler:
                    print(dist)
                    prox_rows.append([j,i])

        if len(prox_rows) == 0: print('None to combine')
                    
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
            prox_rows = combined.copy()
            
        '''Use close-sets to combine dataframe rows and retain all site info'''
        s = '|'
        new = []
        
        for i, rows in enumerate(prox_rows):
            if len(rows) == 0: continue
        
            new.append(self.at_cols.loc[rows[0], :])
            new[i].at['u'] = np.average([self.at_cols.at[ind, 'u'] 
                                         for ind in rows])
            new[i].at['v'] = np.average([self.at_cols.at[ind, 'v'] 
                                         for ind in rows])
            new[i].at['x'] = np.average([self.at_cols.at[ind, 'x'] 
                                         for ind in rows])
            new[i].at['y'] = np.average([self.at_cols.at[ind, 'y'] 
                                         for ind in rows])
            elem = np.unique([self.at_cols.at[ind, 'elem'] for ind in rows])
            new[i].at['elem'] = s.join(elem)
            site_frac = [self.at_cols.at[ind, 'site_frac'] for ind in rows]
            new[i].at['site_frac'] = s.join(site_frac)
            self.at_cols = self.at_cols.drop(rows, axis=0)
            
        self.at_cols = self.at_cols.append(new)
        self.at_cols = self.at_cols.reset_index(drop=True)
        return self.at_cols
    
    def plot_unit_cell(self, label_by='elem'):
        """Plots the projected unit cell for verification.
        
        Parameters
        ----------
        label_by : str
            The DataFrame column to use for labeling the atom columns in the 
            plot.
            
        Returns
        -------
        None.
            
        """
        
        fig,axs = plt.subplots(ncols=1,figsize=(10,10))
        axs.set_aspect(1)
        p = patches.Polygon(np.array([[0, 0],
                                      [self.a_2d[0,0], 0],
                                      [self.a_2d[0,0]+self.a_2d[0,1], 
                                       self.a_2d[1,1]],
                                      [self.a_2d[0,1], self.a_2d[1,1]]]),
                            fill = False, ec = 'black', lw = 0.5)
        axs.add_patch(p)
        axs.set_facecolor('grey')
        ColList = np.sort(pd.unique(self.at_cols[label_by])).tolist()
        cmap = plt.get_cmap('RdYlGn')
        
        color_code = {k:v for v, k in enumerate(
            np.sort(self.at_cols.loc[:, label_by].unique()))}
        color_list = [color_code[site] for site in 
                      self.at_cols.loc[:, label_by]]
    
        cmap = plt.cm.RdYlGn
        
        axs.scatter(self.at_cols.loc[:, 'x'], self.at_cols.loc[:, 'y'],
                    c=color_list, s=300, cmap=cmap, zorder=2)
        
        for ind in self.at_cols.index:
            axs.annotate(rf'${ self.at_cols.at[ind, label_by] }$',
                          (self.at_cols['x'][ind] + 1/8, 
                           self.at_cols['y'][ind] + 1/8), 
                          fontsize=20)
        axs.set_xticks([])
        axs.set_yticks([])
        axs.set_title('Projected Unit Cell', fontdict = {'fontsize' : 20,
                                                         'color' : 'red'})
