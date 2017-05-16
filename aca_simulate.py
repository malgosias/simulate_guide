import numpy as np
from Quaternion import Quat
from agasc.agasc import get_agasc_cone
from mica.archive.aca_dark.dark_cal import get_dark_cal_image
from chandra_aca.aca_image import ACAImage
from Ska.quatutil import yagzag2radec, radec2yagzag
from chandra_aca.transform import pixels_to_yagzag, yagzag_to_pixels, mag_to_count_rate, count_rate_to_mag
import datetime

GAIN = 5. # e-/ADU
FWHM = 1.8 # FWHM of a typical ACA star in px
INTEG = 1.696 # sec

def as_array(attr, imgs):
    out = np.array([getattr(img, attr) for img in imgs])
    return out

    
class ImgList(list):
    def __init__(self, imgs):
        
        super(ImgList, self).__init__(imgs)

        self.times = as_array('TIME', self)
        self.row0s = as_array('row0', self)
        self.col0s = as_array('col0', self)
        self.imgraws = as_array('IMGRAW', self)
        self.bgdavgs = as_array('BGDAVG', self)
        self.funcs = as_array('IMGFUNC1', self)
        self.stats = as_array('IMGSTAT', self)
        self.img_sizes = as_array('IMGSIZE', self)

        self.rows = np.zeros_like(self.times) - 511
        self.cols = np.zeros_like(self.times) - 511
        self.img_sums = np.zeros_like(self.times)
        

    def __getattr__(self, attr):
        return np.array([getattr(img, attr) for img in self])
    
    
    def update_image(self, ii, row0, col0, imgsize):
        
        readout = ACAImage(row0=row0, col0=col0, shape=(imgsize, imgsize))
        imgraw = self[ii][readout]

        bgdimg = imgraw
        if imgsize == 6:
            bgdimg = self[ii][ACAImage(row0=row0 - 1, col0=col0 - 1, shape=(8, 8))]

        bgdavg = calc_bgd(bgdimg)
        img = imgraw - bgdavg
        
        if imgsize == 6:
            img[[0, 0, 5, 5], [0, 5, 0, 5]] = 0.0  # Mouse-bitten
                
        # The 0.5 is a bit of a fudge that I don't quite understand...
        rw, cw = np.mgrid[0:imgsize, 0:imgsize] + 0.5

        if self.funcs[ii] in (1, 3):
            self.img_sums[ii] = norm = np.sum(img)
            self.cols[ii] = np.sum(cw * img) / norm + col0
            self.rows[ii] = np.sum(rw * img) / norm + row0
            
        self[ii].IMGSIZE = imgsize
        self[ii].BGDAVG = bgdavg
        self[ii].IMGRAW = imgraw
        self[ii].col0 = col0
        self[ii].row0 = row0

        self.img_sizes[ii] = imgsize
        self.bgdavgs[ii] = bgdavg
        self.col0s[ii] = col0
        self.row0s[ii] = row0
            
        return


def simulate_guide(quat, yag=0., zag=0., mag=10.0, maxmag=13.0, dither=None,
                   imgsize=6, t_ccd=None, dark=None, select='before', nframes=1000,
                   radius=40., delay=0):
    """
    :quat: initial (catalog) ACA attitude
    :yag, zag, mag: star yag, zag (arcsec), mag (using agasc for mag now)
    :dither: dict with keys for dither y, z ampl (arcsec), period (sec),
             phase (radians) defaults to normal ACIS params.
    :imgsize: size of the ACA readout window in pixels (4, 6 or 8)
    :t_ccd: CCD temperature in degC
    :dark: either a date (in which case it picks the nearest actual dark cal) 
           or a 1024x1024 image (e-/sec).  If image is provided then 't_ccd'
           is ignored.  Default is to use the most recent dark cal.
    :param select: ACA DCC selection for simulated background (before|nearest|after)
    :nframes: number of time frames
    :param radius: spoiler stars within radius will be included in the
                   simulated star field. Size of the simulated star field
                   sz (in px) is derived as the sum of
                   - 2 x radius,
                   - 8 px to properly simulate spoiler stars with separation
                     equal to radius
                   - 6 px due to dither
    :param date_agasc: date for agasc proper motion correction
    """
    
    kwargs = {'dither': dither, 'imgsize': imgsize, 't_ccd': t_ccd, 'dark': dark,
              'select': select, 'nframes': nframes, 'radius': radius}
    
    delay = np.int(delay)
    
    stars = simulate_star_field(quat, yag, zag, mag, **kwargs)
    imgs = ImgList(stars['imgs'][delay:])
    
    col0s = np.zeros_like(imgs.col0s)
    row0s = np.zeros_like(imgs.row0s)
    rows, cols = yagzag_to_pixels(stars['true_yags'][delay], stars['true_zags'][delay])
    
    # This introduces 'pointing' errorsL : 2-3px offset between the 'true' and
    # simulated centroids. Check simulate_star_field. Use telemetry row0, col0
    # for the first two images?    
    row0s[0:2] = np.round([rows - imgsize / 2] * 2)
    col0s[0:2] = np.round([cols - imgsize / 2] * 2)
   
    for ii in (0, 1):
        imgs.update_image(ii, row0s[ii], col0s[ii], imgsize)    
    
    n = len(imgs)
    
    for ii in range(1, n):
        img0 = imgs[ii - 1]
        img1 = imgs[ii]
        
        if img0.IMGFUNC1 == 1 and img1.IMGFUNC1 == 1 and ii < n - 1:
            # Predict next IMGCOL0/IMGROW0 using rate
            rate_r = imgs.rows[ii] - imgs.rows[ii - 1]
            rate_c = imgs.cols[ii] - imgs.cols[ii - 1]
            row0s[ii + 1] = np.round(imgs.rows[ii] + rate_r - imgsize / 2)
            col0s[ii + 1] = np.round(imgs.cols[ii] + rate_c - imgsize / 2)
            imgs.update_image(ii + 1, row0s[ii + 1], col0s[ii + 1], imgsize)
            if imgs.img_sums[ii + 1] < mag_to_count_rate(maxmag) * INTEG / GAIN / 2:
                print('RACQ: image intensity below maxmag')
                break
        elif img0.IMGFUNC1 != 1 and img1.IMGFUNC1 == 1:
            # First track after not tracking (typically search)
            row0s[ii] = np.round(imgs.rows[ii] - imgsize / 2)
            col0s[ii] = np.round(imgs.cols[ii] - imgsize /2)
            if ii < n - 1:
                row0s[ii + 1] = row0s[ii]
                col0s[ii + 1] = col0s[ii]
            imgs.update_images(ii + 1, row0s[ii + 1], col0s[ii + 1], imgsize)
        elif img1.IMGFUNC1 == 2 or img1.IMGFUNC1 == 0:
            # Search or Null => IMGCOL0 is -511
            row0s[ii] = -511
            col0s[ii] = -511
            imgs.update_image(ii, row0s[ii], col0s[ii], imgsize)
        elif img0.IMGFUNC1 != 1 and img1.IMGFUNC1 == 3:  
            # Lost => IMGCOL0 is same as last one (image window doesn't move)
            row0s[ii] = row0s[ii - 1]
            col0s[ii] = col0s[ii - 1]
            imgs.update_image(ii, row0s[ii], col0s[ii], imgsize)

    aca_yags, aca_zags = pixels_to_yagzag(imgs.rows, imgs.cols)
    aca_mags = count_rate_to_mag(imgs.img_sums * GAIN / INTEG)
    dyags = aca_yags - stars['true_yags'][delay:]
    dzags = aca_zags - stars['true_zags'][delay:]
    imgraws = as_array('IMGRAW', imgs)

    guide = {'time': imgs.times,
             'row0s': imgs.row0s,
             'col0s': imgs.col0s,
             'stars': imgs.imgraws,
             'imgraws': imgraws,
             'bgdavgs': imgs.bgdavgs,
             'funcs': imgs.funcs,
             'true_yags': stars['true_yags'][delay:],
             'true_zags': stars['true_zags'][delay:],
             'aca_yags': aca_yags,
             'aca_zags': aca_zags,
             'aca_mags': aca_mags,
             'dyags': dyags,
             'dzags': dzags}
             
    return guide
    
    
def simulate_star_field(quat, yag=0., zag=0., mag=10.0, dither=None, imgsize=6,
                        t_ccd=None, dark=None, select='before',
                        nframes=1000, date_agasc=None, radius=40.):
    """
    Simulate star field including stars within a given radius from 
    yag, zag, and dark background. Attitude changes correspond to the
    dither pattern (or OBC attitude solution between start_obcsol,
    stop_obcsol? TBD if needed).
    """
    
    imgs = []
    
    if imgsize not in (4, 6, 8):
        raise ValueError('imgsize not in (4, 6, 8)')

    # Times
    delta_t = {4: 2.05, 6: 2.05, 8: 4.1}
    times = np.arange(nframes) * delta_t[imgsize]
    
    # Background
    if isinstance(dark, (np.ndarray, list, tuple)):
        if np.shape(dark) != (1024, 1024):
            raise TypeError('Expected dark with size (1024, 1024)')
        dccimg = ACAImage(dark, row0=-512, col0=-512)
    else:
        try:
            dccimg = get_dark_cal_image(date=dark, select=select,
                                        t_ccd_ref=t_ccd, aca_image=True)
        except TypeError as e:
            print("{}. Using today's date".format(e))
            dccimg = get_dark_cal_image(datetime.datetime.now(), select=select,
                                        t_ccd_ref=t_ccd, aca_image=True)
    
    # Dither
    if dither is None:
        dither = {'dither_y_amp': 8., 'dither_z_amp': 8.,
                  'dither_y_period': 1000., 'dither_z_period': 707.,
                  'dither_y_phase': 0., 'dither_z_phase': 0.}

    # Yags, zags of the star, they follow the dither pattern
    yaw = calc_dither(times,
                      dither['dither_y_amp'],
                      dither['dither_y_period'],
                      dither['dither_y_phase'])
    pitch = calc_dither(times,
                        dither['dither_z_amp'],
                        dither['dither_z_period'],
                        dither['dither_z_phase'])
    
    true_yags, true_zags = yag + yaw, zag + pitch

    # Corresponding rows, cols of the star centroid
    true_rows, true_cols = yagzag_to_pixels(true_yags, true_zags)

    # Quaternions
    quats = [quat]    
    for ii in range(1, nframes):
        # What about droll?
        dr = 0.
        dy = yaw[ii] - yaw[ii - 1]
        dp = pitch[ii] - pitch[ii - 1]
        # Sign? -dy, -dp agrees with data but I don't understand why
        # -dp because pitch ~ -DEC
        # why -dy???
        dq = Quat([dy / 3600., -dp / 3600., dr / 3600.])
        quats.append(quats[ii - 1] * dq)        
        
    # Corresponding ras, decs of the star centroid
    true_ras = []
    true_decs = []
    for y, z, q in zip(true_yags, true_zags, quats):
        r, d = yagzag2radec(y / 3600., z / 3600., q)
        true_ras.append(r)
        true_decs.append(d)

    # Size of the simulated star field
    sz = np.int(np.round(2 * radius / 5. + 8 + 6)) # px
    
    # Define ACAImage corresponding to the simulated CCD region
    imgrow0, imgcol0 = np.round(true_rows[0] - sz / 2), np.round(true_cols[0] - sz / 2)
    section = ACAImage(np.zeros((sz, sz)), row0=imgrow0, col0=imgcol0)
    
    for ii in range(nframes):

        # Initate with DCC background
        star_field = dccimg[section] * INTEG / GAIN # convert from e-/sec to AD counts
        # Find spoiler stars
        stars = get_agasc_cone(true_ras[ii], true_decs[ii], radius=radius / 3600.)
        for star in stars:
            img = simulate_star(quats[ii], star['RA'], star['DEC'], star['MAG_ACA'],
                                row0=imgrow0, col0=imgcol0, sz=sz)
            star_field = star_field + img

        meta = {'TIME': times[ii],
                'IMGROW0': imgrow0,
                'IMGCOL0': imgcol0,
                'IMGRAW': ACAImage(star_field, row0=imgrow0, col0=imgcol0),
                'BGDAVG': 0.,
                'IMGFUNC1': 1,
                'IMGSTAT': 0.,
                'IMGSIZE': sz,
                'INTEG': INTEG}        
        
        img = ACAImage(star_field, meta=meta)
        imgs.append(img)

    stars = {'imgs': imgs, 'quats': quats,
             'true_yags': true_yags, 'true_zags': true_zags,
             'true_ras': true_ras, 'true_decs': true_decs,
             'true_rows': true_rows, 'true_cols': true_cols}
        
    return stars


def get_aca_l0_img_list(imgs):
    return ImgList(imgs)


def calc_bgd(img):
    row = [0, 0, 0, 0, 7, 7, 7, 7]
    col = [0, 1, 6, 7, 0, 1, 6, 7]
    vals = np.array(img[row, col])
    while True:
        avg = np.round(np.nanmean(vals))
        sigma = max(avg * 1.5, 10.0)
        dev = np.abs(vals - avg)
        imax = np.nanargmax(dev)
        if dev[imax] > sigma:
            vals[imax] = np.nan
        else:
            break
    return avg#, ~np.isnan(vals)


def calc_dither(times, amp, period, phase):
    """
    Return dither pattern computed over the time
    defined with ndarray times.
    """
    return amp * np.sin(2 * np.pi * times / period + 2 * np.pi * phase)


def simulate_star(quat, ra, dec, mag, row0, col0, sz):
    """
    Simulate a 2-d Gaussian star in a readout window defined by
    row0, col0 and sz. Star's location and magnitude are determined by quat, ra, dec,
    and mag.
    :quat: attitude quaternion
    :ra, dec, mag: RA, DEC and magnitude of the star
    :row0, col0, sz: Coordinates of the lower left corner of the readout window (in pixels)
                     and the size of the readout window (6 or 8 pixels)
    """

    yag, zag = radec2yagzag(ra, dec, quat)
    row, col = yagzag_to_pixels(yag * 3600, zag * 3600)

    # Gaussian model, grid definition with (0, 0) in the center to enable count normalization
    halfsize = np.int(sz / 2)
    # Without -0.5 simulated centroids are off from telemetry by 0.5px
    r, c = np.mgrid[-halfsize:halfsize, -halfsize:halfsize] - 0.5
    sigma = FWHM / (2. * np.sqrt(2. * np.log(2.)))
    # Offset are ~zero for the main star
    roff = row - row0 - halfsize
    coff = col - col0 - halfsize
    g = np.exp(-((r - roff)**2  / sigma**2 + (c - coff)**2 / sigma**2) / 2.)

    # Mag to counts conversion
    counts =  mag_to_count_rate(mag) * INTEG / GAIN

    # Normalize to counts, 6x6 contains all the counts, ok?
    r1 = np.int(np.round(roff - 3 - halfsize))
    r2 = np.int(np.round(roff + 3 - halfsize))
    c1 = np.int(np.round(coff - 3 - halfsize))
    c2 = np.int(np.round(coff + 3 - halfsize))
    
    g = counts * g / g[r1: r2, c1: c2].sum()

    # Simulate star
    star = ACAImage(np.random.normal(g), row0=row0, col0=col0)

    return star

