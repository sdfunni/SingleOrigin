from scipy.ndimage import binary_erosion
import time
from scipy.interpolate import SmoothBivariateSpline

p0 = acl.at_cols.loc[:, 'x_fit':'bkgd_int'].to_numpy()
p0 = np.delete(p0, 3, 1)
p0[:, 4] = np.radians(p0[:, 4])

p_max = np.max(p0[:, :-1], axis=0)

p_max[:2] = [99, 99]
p_max[-1] = 0
[x0, y0, sig_maj, sig_rat, ang, A] = p_max
y, x = np.indices((acl.image.shape))
x = x.ravel()
y = y.ravel()
xy = np.vstack((x,y))

z = gaussian_2d(x, y, x0, y0, sig_maj, sig_rat, ang, A=1)
z = np.reshape(np.where(z < 0.01, z, 0), (200,200))
mask = np.where(z, 1, 0)
edge = mask - np.where(binary_erosion(mask, border_value=1), 1, 0)
edge_ind = np.vstack(((np.take(x, np.nonzero(edge.ravel()))),
                      np.take(y, np.nonzero(edge.ravel())))).T
tol = np.ceil(np.max(np.linalg.norm(edge_ind - [99, 99], axis=1)))

#%%
def multi_gauss_ss(self, p0):
    p0 = np.reshape(p0, (p0.shape[0]/7, 7))
    p0[:, 3] = p0[:, 2] / p0[:, 3]
    y, x = np.indices((acl.image.shape))
    
    z = (acl.image * acl.all_masks).ravel()
    x = x.ravel()
    y = y.ravel()
    
    unmasked_data = np.nonzero(z)
    x = np.take(x, unmasked_data)
    y = np.take(y, unmasked_data)
    z = np.take(z, unmasked_data)
    
    #Model background with bivariate spline
    bkgd = SmoothBivariateSpline(p0[:, 0], p0[:, 1], p0[:, 6], 
                               kx=5, ky=5)
    
    z -= bkgd.__call__(x, y, grid=False)
    
    for atom in p0:
        [x0, y0, sig_maj, sig_min, ang, A] = atom[:-1]
        
        test = np.where(((x > x0 - tol) & (x < x0 + tol) & 
                         (y > y0 - tol) & (y < y0 + tol)))
        x_ = np.take(x, test)
        y_ = np.take(y, test)
        z_ = np.take(z, test)
        
        z_ -= A*np.exp(-1/2*(((np.cos(ang) * (x_ - x0) 
                               + np.sin(ang) * (y_ - y0)) / sig_maj)**2
                             +((-np.sin(ang) * (x_ - x0) 
                                + np.cos(ang) * (y_ - y0)) / sig_min)**2))
        
        np.put(z, test, z_)
    
    R_sq = z @ z.T
    R_sq = R_sq[0,0]
    
    return R_sq

#%%
z = copy.deepcopy(acl.image)
p0 = acl.at_cols.loc[:, 'x_fit': 'bkgd_int'].to_numpy()

tol=20
for atom in p0:
    [x0, y0, sig_maj, sig_min, sig_rat, ang, A, I0] = atom
    
    test = np.where(((x > x0 - tol) & (x < x0 + tol) & 
                     (y > y0 - tol) & (y < y0 + tol)))
    x_ = np.take(x, test)
    y_ = np.take(y, test)
    z_ = np.take(z, test)
    
    z_ -= A*np.exp(-1/2*(((np.cos(ang) * (x_ - x0) 
                           + np.sin(ang) * (y_ - y0)) / sig_maj)**2
                         +((-np.sin(ang) * (x_ - x0) 
                            + np.cos(ang) * (y_ - y0)) / sig_min)**2))
    
    np.put(z, test, z_)




#%%
xy = np.ravel_multi_index((y,x), acl.image.shape)
z_recon = np.zeros(acl.image.shape).ravel()
np.put(z_recon, xy, bkgd.flatten())
#%%
plt.figure(figsize=(20,20))

plt.imshow(z)
