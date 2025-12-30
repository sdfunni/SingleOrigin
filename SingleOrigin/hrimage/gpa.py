"""Module for Fourier phase analysis of high resolution STEM images."""

import copy

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import cmocean

import numpy as np

from scipy.fft import (
    fft2,
    ifft2,
    fftshift,
)

from scipy.signal import medfilt2d

from SingleOrigin.utils.peakfit import (
    get_feature_size,
    watershed_segment,
    gaussian_2d,
    fit_gaussians,
)

from SingleOrigin.utils.plot import (
    pick_points,
    quickplot,
)

from SingleOrigin.utils.image import lowpass_filter
from SingleOrigin.utils.fourier import hann_2d

# %%


def phase_ref(xy, shape):
    """
    Create cosine and sine waves corresponding to a specified FFT peak.

    Parameters
    ----------
    xy : array
        Peak position (x, y) in the shifted (zero frequency centered) FFT.
    shape : tuple
        Shape of the image (height, width).

    Returns
    -------
    cosRef, sinRef : ndarrays
        Cosine and sine reference waves with same shape as 'shape'.
    """

    # Find k vector for the corresponding peak
    kxy = (2 * np.pi * (xy - np.flip(shape)/2) /
           np.flip(shape))[:, None, None]

    # Calculate real space coordinate vectors for image pixels
    rxy = np.flip(np.indices(shape), axis=0)

    # Take dot product of k & r
    theta = np.sum(kxy * rxy, axis=0)

    # Get cos and sin waves
    cosRef = np.cos(theta)
    sinRef = np.sin(theta)

    return cosRef, sinRef


class FourierAnalysis():
    """
    Class for running quantitative Fourier analysis of an image. Creating a
    class object allows retention and organization of data obtained during
    some function calls that otherwise needs to be managed by the user in each
    script.

    Parameters
    ----------
    image : ndarray
        The STEM image to analize.

    Attributes
    ----------
    h, w : ints
        The height and width of the image.

    Methods
    -------
    pick_fft_peaks

    phase_lock_in

    """

    def __init__(
        self,
        image,
        pixelSize=None,
        pixelUnit=None,
        roi=None,
    ):

        crop = np.array(image.shape)//2 * 2
        self.image = copy.deepcopy(image[:crop[0], :crop[1]])

        self.h, self.w = image.shape
        self.pixelSize = pixelSize
        self.pixelUnit = pixelUnit
        if roi is None:
            self.roi = np.ones(image.shape)
        else:
            self.roi = roi

    def pick_fft_peaks(
            self,
            n_peaks=1,
            zoom=5,
            thresh=0.25,
            selection_roi=None,
            mask_for_speed=True,
            sigma=None,
    ):
        """
        Pick FFT peaks for subsequent Fourier analysis.

        Parameters
        ----------
        image : 2d array
            The image to analize.

        n_peaks : int
            The number of peaks to choose in the FFT. A phase map will be
            generated for each peak picked.

        zoom : scalar
            Factor to zoom on the FFT when choosing peaks.
            Default: 5.

        thresh : scalar
            Thresholding factor for masking the peak area for Gaussian fitting
            of chosen FFT peaks.
            Default: 0.25.

        selection_roi : binary array of same shape as self.image
            The ROI for acquiring the FFT and measuring peak positions.

        mask_for_speed : bool
            Whether to mask the FFT outside 1024x1024 pixels to speed up peak
            detection (which may be slow for very large images).

        Returns
        -------
        None.

        """

        # Get power spectrum of image
        if selection_roi is None:
            # Set whole image as ROI
            selection_roi = np.ones(self.image.shape)
        else:
            # Make sure ROI is binary 0's and 1's
            selection_roi = np.where(selection_roi > 0, 1, 0)

        self.window_size = np.min(self.image.shape) / zoom

        hann = hann_2d(self.image.shape)
        self.pwrfft = np.abs(fftshift(fft2(self.image * selection_roi * hann)))

        # Crop FFT for large image sizes: speed up peak detection
        origin = np.array([self.h, self.w]) // 2
        if mask_for_speed:
            if np.max([self.h, self.w]) > 1024:
                fftmask = np.zeros((self.h, self.w))
                fftmask[origin[0] - 512: origin[0] + 512,
                        origin[1] - 512: origin[1] + 512] = 1
            else:
                fftmask = None

        # Find FFT peaks
        pwrfft_centermasked = copy.deepcopy(self.pwrfft)
        pwrfft_centermasked[origin[0] - 10: origin[0] + 10,
                            origin[1] - 10: origin[1] + 10] = 0
        if sigma is None:
            self.sigma = get_feature_size(pwrfft_centermasked) * 5
        else:
            self.sigma = sigma

        print('finding peaks')
        masks, num_masks, _, peaks = watershed_segment(
            self.pwrfft,
            roi=fftmask,
            sigma=self.sigma,
            filter_type='log',
            # bkgd_thresh_factor=thresh,
            peak_bkgd_thresh_factor=thresh,
            min_dist=2*self.sigma,
            buffer=10
        )

        self.peak_masks = masks

        peaks = peaks.loc[:, 'x':'y'].to_numpy()
        print('Select peaks...')

        self.xy_selected = pick_points(
            n_picks=n_peaks,
            image=self.pwrfft,
            xy_peaks=peaks,
            window_size=self.window_size,
            quickplot_kwargs={'scaling': 'log'}
        )

    def phase_lock_in(
            self,
            maskSigma,
            lockinSigma,
            add_phase=0,
            refxy=None,
    ):
        """
        Perform phase lock-in analysis of an atomic resolution image for peaks
        previously selected using the pick_fft_peaks() method. Essentially this
        makes a map of relative phase shift of the selected frequency across an
        image.

        Parameters
        ----------
        maskSigma : scalar
            The sigma for guassian masking of FFT peak(s).

        lockinSigma : scalar
            The real space Gaussian filter width used for smoothing the
            resulting phase shift map. (i.e. the real space coarsening.)

        add_phase : scalar
            Phase shift in radians to add to the resulting phase map. This does
            not change the phase map qualitiatively but will shift the colors
            displayed. Adding a phase shift may be preferred for phase maps
            with relatively flat contrast to select a specific part of the
            colormap for asthetic purposes.

        ref_xy : ndarray or None.
            Reference vector for calculating the phase if the fitted FFT peak
            is not desired for some reason. If not the fitted FFT peak will be
            used (this should nearly always be the case).
            USE WITH CAUTION!
            default: None.

        Returns
        -------
        None
        """

        # Fit peaks
        self.xy_selected = np.array(self.xy_selected)
        # masks_picked = [self.peak_masks[y, x] for x, y in self.xy_selected]
        masks_picked = [self.peak_masks[y, x] for x, y in
                        np.around(self.xy_selected).astype(int)]

        xy = np.flip(np.indices(self.image.shape), axis=0)

        fitargs = [[
            xy[:, self.peak_masks == mask_num],
            self.pwrfft[self.peak_masks == mask_num],
            np.concatenate([
                self.xy_selected[i],
                [1, np.max(self.pwrfft[self.peak_masks == mask_num]), 0]
            ]),
        ] for i, mask_num in enumerate(masks_picked)]

        fits_xy = np.array([
            fit_gaussians(*args).squeeze()[:2] for args in fitargs
        ])

        realMaps = []
        amplitudeMaps = []
        phaseMaps = []
        qxy = []
        masks = []

        if refxy is None:
            refxy = fits_xy

        # Calculate phase and amplitude maps for each peak
        for i, peak in enumerate(fits_xy):

            cosRef, sinRef = phase_ref(refxy[i], (self.h, self.w))

            self.refs = cosRef

            # Get decentered FFT peak mask
            mask = fftshift(gaussian_2d(
                xy[0],
                xy[1],
                x0=peak[0],
                y0=peak[1],
                sig_maj=maskSigma,
                sig_rat=1,
                ang=0,
                A=1,
                b=0,
            ))

            im_fftfiltered = ifft2(fft2(self.image) * mask)

            # Get lowpass filtered cos and sin compoenents of the frequency
            cosComp = np.real(lowpass_filter(
                cosRef * im_fftfiltered,
                sigma=lockinSigma,
                get_abs=False,
            ))
            sinComp = np.real(lowpass_filter(
                sinRef * im_fftfiltered,
                sigma=lockinSigma,
                get_abs=False,
            ))

            # Get the phase angle

            phaseMaps += [(np.arctan2(sinComp, cosComp) + add_phase
                           ) % (2*np.pi)]
            realMaps += [np.real(im_fftfiltered)]
            amplitudeMaps += [np.abs(im_fftfiltered)]
            qxy += [peak - np.array([self.h, self.w]) / 2]
            masks += [mask]

        # if self.xy_selected.shape[0] == 1:
        self.phaseMaps = phaseMaps * self.roi
        self.realMaps = realMaps
        self.amplitudeMaps = amplitudeMaps
        self.qxy = qxy
        self.masks = masks

    def multipeak_phase_lock_in(
            self,
            maskSigma,
            lockinSigma,
            add_phase=0,
            refxy=None,
    ):
        """
        Perform phase lock-in analysis of an atomic resolution image for peaks
        previously selected using the pick_fft_peaks() method. Essentially this
        makes a map of relative phase shift of the selected frequency across an
        image.

        Parameters
        ----------
        maskSigma : scalar
            The sigma for guassian masking of FFT peak(s).

        lockinSigma : scalar
            The real space Gaussian filter width used for smoothing the
            resulting phase shift map. (i.e. the real space coarsening.)

        add_phase : scalar
            Phase shift in radians to add to the resulting phase map. This does
            not change the phase map qualitiatively but will shift the colors
            displayed. Adding a phase shift may be preferred for phase maps
            with relatively flat contrast to select a specific part of the
            colormap for asthetic purposes.

        ref_xy : ndarray or None.
            Reference vector for calculating the phase if the fitted FFT peak
            is not desired for some reason. If not the fitted FFT peak will be
            used (this should nearly always be the case).
            USE WITH CAUTION!
            default: None.

        Returns
        -------
        None
        """

        # Fit peaks
        self.xy_selected = np.array(self.xy_selected)
        # masks_picked = [self.peak_masks[y, x] for x, y in self.xy_selected]
        masks_picked = [self.peak_masks[y, x] for x, y in
                        np.around(self.xy_selected).astype(int)]

        xy = np.flip(np.indices(self.image.shape), axis=0)

        fitargs = [[
            xy[:, self.peak_masks == mask_num],
            self.pwrfft[self.peak_masks == mask_num],
            np.concatenate([
                self.xy_selected[i],
                [1, np.max(self.pwrfft[self.peak_masks == mask_num]), 0]
            ]),
        ] for i, mask_num in enumerate(masks_picked)]

        fits_xy = np.array([
            fit_gaussians(*args).squeeze()[:2] for args in fitargs
        ])

        realMaps = []
        amplitudeMaps = []
        phaseMaps = []
        qxy = []
        masks = []

        if refxy is None:
            refxy = fits_xy

        # Calculate phase and amplitude maps for each peak

        cosRef, sinRef = phase_ref(np.mean(fits_xy, axis=0),
                                   (self.h, self.w))

        self.refs = cosRef

        # Get decentered FFT peak mask
        mask = fftshift(gaussian_2d(
            xy[0],
            xy[1],
            x0=fits_xy[:, 0][:, None, None],
            y0=fits_xy[:, 1][:, None, None],
            sig_maj=maskSigma,
            sig_rat=1,
            ang=0,
            A=1,
            b=0,
        ))

        im_fftfiltered = ifft2(fft2(self.image) * mask)

        # Get lowpass filtered cos and sin compoenents of the frequency
        cosComp = np.real(lowpass_filter(
            cosRef * im_fftfiltered,
            sigma=lockinSigma,
            get_abs=False,
        ))
        sinComp = np.real(lowpass_filter(
            sinRef * im_fftfiltered,
            sigma=lockinSigma,
            get_abs=False,
        ))

        # Get the phase angle

        phaseMaps += [(np.arctan2(sinComp, cosComp) + add_phase
                       ) % (2*np.pi)]
        realMaps += [np.real(im_fftfiltered)]
        amplitudeMaps += [np.abs(im_fftfiltered)]
        qxy += [np.mean(fits_xy, axis=0) - np.array([self.h, self.w]) / 2]
        masks += [mask]

        # if self.xy_selected.shape[0] == 1:
        self.phaseMaps = phaseMaps * self.roi
        self.realMaps = realMaps
        self.amplitudeMaps = amplitudeMaps
        self.qxy = qxy
        self.masks = masks

    def plot_phase_lock_in_maps(
            self,
            peak_ind=0,
            figax=True,
            arrangement='square',
            plot_real=True,
    ):
        """
        Make a plot figure for a phase lock-in map.

        Parameters
        ----------
        peak_ind : int
            Index of the desired phase map. Indexing is in the order in which
            FFT peaks were chosen.
            Default: 0.

        figax : bool
            Whether to return the matplotlib figure and axes objects.

        arrangement : str
            Arrangement of the plots in the figure: 'square' (2x2),
            'horizontal', or 'vertical'. Square only applies if plot_real is
            True, otherwise defaulting to horizontal.
            Default: 'square'.

        plot_real : bool
            If True, will plot the real component of the FFT filtered image.
            Otherwise only the original image, FFT with gaussian mask, and
            phase map overlaid on the image. The real component map is useful
            for visualizing dislocations in the structure.

        Returns
        -------
        fig, axs : Matplotlib Figure and array of Axes objects (optionally).

        """

        # TODO: add a real space coarsening circle.

        if plot_real:
            if arrangement == 'square':
                subplots = (2, 2)
                figsize = (10, 10)
            elif arrangement == 'horizontal':
                subplots = (1, 4)
                figsize = (20, 5)
            elif arrangement == 'vertical':
                subplots = (4, 1)
                figsize = (5, 20)

        else:
            if np.isin(arrangement, ['horizontal', 'square']):
                subplots = (1, 3)
                figsize = (15, 5)
            elif arrangement == 'vertical':
                subplots = (3, 1)
                figsize = (5, 15)

        fig, axs = plt.subplots(*subplots,
                                # sharex=True, sharey=True,
                                figsize=figsize)

        axs = axs.flatten()

        # Plot FFT with Guassian mask overlaid
        quickplot(
            np.abs(fftshift(fft2(self.image))),
            scaling='log',
            cmap='grey',
            figax=axs[0]
        )

        axs[0].imshow(fftshift(self.masks[peak_ind]),
                      cmap='inferno', alpha=0.5)

        axs[0].set_ylim(
            bottom=self.h//2 + self.window_size/2,
            top=self.h//2 - self.window_size/2
        )

        axs[0].set_xlim(
            left=self.w//2 - self.window_size/2,
            right=self.w//2 + self.window_size/2
        )

        # Plot image
        quickplot(
            self.image,
            cmap='grey',
            figax=axs[1],
            pixelSize=self.pixelSize,
            pixelUnit=self.pixelUnit,
            scalebar_len=2,
        )

        # Plot phase shift map
        quickplot(self.image, cmap='grey', figax=axs[2])
        axs[2].set_title('Phase')
        axs[2].imshow(self.phaseMaps[peak_ind],
                      cmap=cmocean.cm.phase, alpha=0.8,
                      vmin=0, vmax=2*np.pi
                      )

        axs[2].contour(
            self.phaseMaps[peak_ind],
            levels=np.arange(0, 2*np.pi, np.pi/2),
            colors='black',
            linewidths=0.5
        )
        axs[2].sharex(axs[1])
        axs[2].sharey(axs[1])

        if plot_real:
            # Plot real component of the FFT (lattice planes)
            axs[3].set_title('Real comp.')

            quickplot(self.realMaps[peak_ind], figax=axs[3])
            axs[3].sharex(axs[1])
            axs[3].sharey(axs[1])

        if figax is True:
            return fig, axs

    def get_strain(self, plot=True, figax=False):
        """
        Get strain map from a locked-in phase maps. This requires that exactly
        two FFT peaks were chosen for the phase lock-in analysis. It is also
        necessary that the chosen peaks be nearly orthogonal.

        Parameters
        ----------
        plot : bool
            Whether to plot the strain maps after calculation.

        figax : bool
            Wether to return the figure and axes objects for modification by
            the user.
            Default: False

        Returns
        -------
        None.

        """
        # TODO : Generalize for non-orthogonal vectors and lattice not aligned
        # to scan axes.

        self.strain_maps = {}
        phase = self.phaseMaps[0]
        phasegrad = np.flip(np.array(np.gradient(phase)), axis=0)
        qxy1 = self.qxy[0]

        eps_xx = np.sum(
            (qxy1 / np.linalg.norm(qxy1)**2)[:, None, None] * phasegrad,
            axis=0) * self.h / (2 * np.pi)

        qxy1_ = np.flip(qxy1) * [1, -1]
        eps_xy = 0.5 * np.sum(
            (qxy1_ / np.linalg.norm(qxy1_)**2)[:, None, None] * phasegrad,
            axis=0) * self.h / (2 * np.pi)

        phase = self.phaseMaps[1]
        phasegrad = np.flip(np.array(np.gradient(phase)), axis=0)
        qxy2 = self.qxy[1]

        eps_yy = np.sum(
            (qxy2 / np.linalg.norm(qxy2)**2)[:, None, None] * phasegrad,
            axis=0) * self.h / (2 * np.pi)

        qxy2_ = np.flip(qxy2) * [1, -1]
        eps_yx = 0.5 * np.sum(
            (qxy2_ / np.linalg.norm(qxy2_)**2)[:, None, None] * phasegrad,
            axis=0) * self.h / (2 * np.pi)

        self.strain_maps['exx'] = medfilt2d(eps_xx, kernel_size=5)
        self.strain_maps['exy'] = medfilt2d(eps_xy, kernel_size=5)
        self.strain_maps['eyx'] = medfilt2d(eps_yx, kernel_size=5)
        self.strain_maps['eyy'] = medfilt2d(eps_yy, kernel_size=5)
        self.strain_maps['theta'] = np.degrees(
            0.5*(self.strain_maps['exy'] -
                 self.strain_maps['eyx'])
        )

        if plot:
            comps = ['im', 'exx', 'eyy', 'exy', 'theta']
            plotinds = np.array([[0, 0], [0, 1], [0, 2], [3, 1], [3, 2]])
            labels = {
                'exx': r'$\epsilon _{xx}$',
                'eyy': r'$\epsilon _{yy}$',
                'exy': r'$\epsilon _{xy}$',
                'theta': r'$\theta$'
            }

            fig = plt.figure(layout='tight', figsize=(15, 12))
            gs = fig.add_gridspec(
                nrows=5,
                ncols=3,
                width_ratios=[1, 1, 1],
                height_ratios=[1, 0.05, 0.05, 1, 0.05],
                # wspace=0.05,
                # hspace=0.1,
            )
            axs = []
            plots = []
            cbarlist = []
            cbaraxs = []

            for i, comp in enumerate(comps):
                if i > 0:
                    axs += [fig.add_subplot(
                        gs[*plotinds[i]],
                        sharex=axs[0],
                        sharey=axs[0]
                    )]
                else:
                    axs += [fig.add_subplot(gs[*plotinds[i]])]
                if comp == 'im':
                    quickplot(
                        self.image,
                        cmap='grey',
                        figax=axs[0],
                        pixelSize=self.pixelSize,
                        pixelUnit=self.pixelUnit,
                        scalebar_len=5,
                    )
                else:
                    if comp != 'theta':
                        factor = 100
                    else:
                        factor = 1
                    plots += [quickplot(
                        self.strain_maps[comp]*factor,
                        cmap='bwr',
                        vmin=-5,
                        vmax=5,
                        figax=axs[i],
                        returnplot=True,
                    )]

                    pad_ = np.min([self.w, self.h]) * 0.05
                    text = axs[i].text(
                        pad_, pad_,
                        labels[comp], ha='left', va='top', size=20,
                        bbox=dict(facecolor='white',
                                  alpha=0.8, lw=0)
                    )

                    # text_bbox =
                    text.get_bbox_patch()

                    subgs = gridspec.GridSpecFromSubplotSpec(
                        1, 4, subplot_spec=gs[*(plotinds[i] + [1, 0])])
                    cbaraxs += [fig.add_subplot(subgs[1:3])]
                    # cbarax = fig.add_subplot(gs[*(plotinds[i] + [1, 0])])
                    units = r'$\circ$' if comp == 'theta' else '%'
                    cbarlist += [plt.colorbar(
                        plots[-1],
                        cax=cbaraxs[-1],
                        orientation='horizontal',
                        # shrink=0.3,
                        aspect=1,
                        ticks=[-5, 0, 5],
                        pad=0.5,
                        # label=units,
                        use_gridspec=True,
                    )]

                    cbarlist[-1].ax.tick_params(labelsize=12)
                    cbarlist[-1].set_label(
                        label=units, fontsize=16, fontweight='bold')
        if figax:
            return fig, axs
