import numpy as np
from Quaternion import Quat
from agasc.agasc import get_agasc_cone
from mica.archive.aca_dark.dark_cal import get_dark_cal_image
from chandra_aca.aca_image import ACAImage
from Ska.quatutil import yagzag2radec, radec2yagzag
from chandra_aca.transform import pixels_to_yagzag, yagzag_to_pixels, mag_to_count_rate, count_rate_to_mag
import datetime


GAIN = 5. # e-/ADU
FWHM = 2.77 # FWHM of a typical ACA star in px
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


def simulate_guide(quat, yag=0., zag=0., maxmag=13.0, dither=None,
                   imgsize=6, t_ccd=None, dark=None, select='before', nframes=1000,
                   radius=40.):
    """
    :quat: initial (catalog) ACA attitude
    :yag, zag, maxmag: star yag, zag (arcsec), faintest star magnitude
    :dither: dict with keys for dither y, z ampl (arcsec), period (sec),
             phase (radians) defaults to normal ACIS params.
    :imgsize: size of the ACA readout window in pixels (4, 6 or 8)
    :t_ccd: CCD temperature in degC
    :dark: either a date (in which case it picks the nearest actual dark cal) 
           or a 1024x1024 image (e-/sec).  If image is provided then 't_ccd'
           is ignored.  Default is to use the most recent dark cal.
    :select: ACA DCC selection for simulated background (before|nearest|after)
    :nframes: number of time frames
    :radius: in arcsec, simulated star field includes stars within distance
             given with radius from yag, zag
    """

    if imgsize not in (4, 6, 8):
        raise ValueError('imgsize not in (4, 6, 8)')

    # Times
    delta_t = {4: 2.05, 6: 2.05, 8: 4.1}
    times = np.arange(nframes) * delta_t[imgsize]
    
    # Dither
    if dither is None:
        dither = {'dither_y_amp': 8., 'dither_z_amp': 8.,
                  'dither_y_period': 1000., 'dither_z_period': 707.,
                  'dither_y_phase': 0., 'dither_z_phase': 0.}

    # Yaw / Pitch
    yaw = calc_dither(times,
                      dither['dither_y_amp'],
                      dither['dither_y_period'],
                      dither['dither_y_phase'])
    pitch = calc_dither(times,
                        dither['dither_z_amp'],
                        dither['dither_z_period'],
                        dither['dither_z_phase'])
    
    print(dither, yag, zag, quat)
    
    # Fetch spoiler stars within radius from the catalog yag, zag,
    # including the star of interest
    ra, dec = yagzag2radec(yag / 3600., zag / 3600., quat)
    stars = get_agasc_cone(ra, dec, radius=radius/3600.)

    # Simulate star field
    kwargs = {'t_ccd': t_ccd, 'dark': dark, 'select': select}
    star_field = simulate_star_field(stars, quat, yag, zag, radius=1.5*radius, **kwargs)
    
    # Initialize the first two images
    col0s = np.zeros(nframes)
    row0s = np.zeros(nframes)
    img_sums = np.zeros(nframes)
    rows = np.zeros(nframes)
    cols = np.zeros(nframes)
    
    row, col = yagzag_to_pixels(yag, zag)
    row0s[0:2] = np.round([row - imgsize / 2] * 2)
    col0s[0:2] = np.round([col - imgsize / 2] * 2)

    readout = ACAImage(row0=row0s[0], col0=col0s[0], shape=(imgsize, imgsize))
    imgs = [star_field[readout], star_field[readout]]
    star_fields = [star_field, star_field]

    for ii in (0, 1):
        imgs[ii].IMGSIZE = imgsize
        imgs[ii] = subtract_bgd(imgs[ii])
        img_sums[ii], rows[ii], cols[ii] = calc_centroids(imgs[ii])
        imgs[ii].IMGRAW = star_field[readout]

    # Initialize the first two quaternions, true yag/zag coordinates, and ra/dec
    dy, dp = yaw[1] - yaw[0], pitch[1] - pitch[0]
    dq = Quat([dy / 3600., -dp / 3600., 0.])
    quats = [quat, quat * dq]
    true_ras = [ra, ra + dq.ra0]
    true_decs = [dec, dec + dq.dec]
    true_yags = [yag, yag]
    true_zags = [zag, zag]

    counter = 0
    
    for ii in range(1, nframes - 1):
        
        dy = yaw[ii + 1] - yaw[ii]
        dp = pitch[ii + 1] - pitch[ii]
        dq = Quat([dy / 3600., -dp / 3600., 0.])
        quats.append(quats[ii] * dq)

        ra = ra + dq.ra0
        dec = dec + dq.dec
        true_ras.append(ra)
        true_decs.append(dec)

        y, z = radec2yagzag(ra, dec, quats[ii + 1])
        true_yags.append(y * 3600.)
        true_zags.append(z * 3600.)
        
        img0 = imgs[ii - 1]
        img1 = imgs[ii]
        
        if img0.IMGFUNC1 == 1 and img1.IMGFUNC1 == 1 and ii < nframes - 1:
            # Predict next IMGCOL0/IMGROW0 using rate
            rate_r = rows[ii] - rows[ii - 1]
            rate_c = cols[ii] - cols[ii - 1]
            row0s[ii + 1] = np.round(rows[ii] + rate_r - imgsize / 2)
            col0s[ii + 1] = np.round(cols[ii] + rate_c - imgsize / 2)
            readout = ACAImage(row0=row0s[ii + 1], col0=col0s[ii + 1], shape=(imgsize, imgsize))
        elif img0.IMGFUNC1 != 1 and img1.IMGFUNC1 == 1:
            # First track after not tracking (typically search)
            row0s[ii] = np.round(rows[ii] - imgsize / 2)
            col0s[ii] = np.round(cols[ii] - imgsize /2)
            if ii < n - 1:
                row0s[ii + 1] = row0s[ii]
                col0s[ii + 1] = col0s[ii]
            readout = ACAImage(row0=row0s[ii + 1], col0=col0s[ii + 1], shape=(imgsize, imgsize))
        elif img1.IMGFUNC1 == 2 or img1.IMGFUNC1 == 0:
            # Search or Null => IMGCOL0 is -511
            row0s[ii] = -511
            col0s[ii] = -511
            readout = ACAImage(row0=row0s[ii], col0=col0s[ii], shape=(imgsize, imgsize))
        elif img0.IMGFUNC1 != 1 and img1.IMGFUNC1 == 3:  
            # Lost => IMGCOL0 is same as last one (image window doesn't move)
            row0s[ii] = row0s[ii - 1]
            col0s[ii] = col0s[ii - 1]
            readout = ACAImage(row0=row0s[ii], col0=col0s[ii], shape=(imgsize, imgsize))

        star_field = simulate_star_field(stars, quats[-1], true_yags[-1], true_zags[-1], radius=1.5*radius, **kwargs)
        star_fields.append(star_field)
        
        new_img = star_field[readout]
        new_img.IMGSIZE = imgsize
        new_img = subtract_bgd(new_img)
        new_img.IMGRAW = star_field[readout]
        if ii < nframes - 1:
            imgs.append(new_img)        
            img_sums[ii + 1], rows[ii + 1], cols[ii + 1] = calc_centroids(new_img)
            if img_sums[ii + 1] < mag_to_count_rate(maxmag) * INTEG / GAIN / 2:
                #print('RACQ: image intensity below maxmag')
                counter += 1
                if delta_t[imgsize] * counter > 100:
                    # Lost
                    print('star lost')
                    img1.IMGFUNC1 = 2 # ???
                    new_img.IMGFUNC1 = 3
                    counter = 0

    imgs = ImgList(imgs)
    imgs.rows = rows
    imgs.cols = cols
    imgs.img_sums = img_sums
    imgs.times = times        
    aca_yags, aca_zags = pixels_to_yagzag(imgs.rows, imgs.cols)
    aca_mags = count_rate_to_mag(imgs.img_sums * GAIN / INTEG)
    dyags = aca_yags - true_yags
    dzags = aca_zags - true_zags
    imgraws = as_array('IMGRAW', imgs)

    guide = {'time': imgs.times,
             'row0s': imgs.row0s,
             'col0s': imgs.col0s,
             'imgraws': imgs.imgraws,
             'bgdavgs': imgs.bgdavgs,
             'funcs': imgs.funcs,
             'true_yags': true_yags,
             'true_zags': true_zags,
             'true_ras': true_ras,
             'true_decs': true_decs,
             'aca_yags': aca_yags,
             'aca_zags': aca_zags,
             'aca_mags': aca_mags,
             'dyags': dyags,
             'dzags': dzags,
             'stars': star_fields,
             'quats': quats}
             
    return guide    
    
    
def simulate_star_field(stars, quat, yag, zag, radius,
                        t_ccd=None, dark=None, select='before'):
    """
    Simulate star field including stars within a given radius from 
    yag, zag, and dark background. Attitude changes correspond to the
    dither pattern (or OBC attitude solution between start_obcsol,
    stop_obcsol? TBD if needed).

    :stars: agasc stars to be simulated (output of get_agasc_cone)
    :radius: in arcsec
    """
    
    # Dark current background
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
    
    # Size of the simulated star field
    sz = np.int(np.round(2 * radius / 5.)) # px

    # Define ACAImage corresponding to the simulated CCD region
    row, col = yagzag_to_pixels(yag, zag)
    imgrow0, imgcol0 = np.round(row - sz / 2.), np.round(col - sz / 2.)
    section = ACAImage(np.zeros((sz, sz)), row0=imgrow0, col0=imgcol0)
    
    # Initate with DCC background
    star_field = dccimg[section] * INTEG / GAIN # convert from e-/sec to AD counts
    
    for star in stars:
        img = simulate_star(quat, star['RA'], star['DEC'], star['MAG_ACA'],
                            row0=imgrow0, col0=imgcol0, sz=sz)
        star_field = star_field + img        

    meta = {'TIME': 0.,
            'IMGROW0': imgrow0,
            'IMGCOL0': imgcol0,
            'IMGRAW': ACAImage(star_field, row0=imgrow0, col0=imgcol0),
            'BGDAVG': 0.,
            'IMGFUNC1': 1,
            'IMGSTAT': 0.,
            'IMGSIZE': sz,
            'INTEG': INTEG}        

    # This is a large 30x30px region around the catalog position of the star
    star_field = ACAImage(star_field, meta=meta)
        
    return star_field


def calc_dither(times, amp, period, phase):
    """
    Return dither pattern computed over the time
    defined with ndarray times.
    """
    return amp * np.sin(2 * np.pi * times / period + 2 * np.pi * phase)


def simulate_star(quat, ra, dec, mag, row0, col0, sz):
    """
    Simulate a 2-d Gaussian star in a readout window defined by
    row0, col0 and sz. Star's location and magnitude are given
    by quat, ra, dec, and mag.
    
    :quat: attitude quaternion
    :ra, dec, mag: RA, DEC and magnitude of the star
    :row0, col0: Coordinates of the lower left corner (in pixels)
    :sz: Size of the readout window (6 or 8 pixels)
    """

    yag, zag = radec2yagzag(ra, dec, quat)
    row, col = yagzag_to_pixels(yag * 3600, zag * 3600)
    halfsize = np.int(sz / 2)

    # Grid centered at (0, 0)
    # Without -0.5 simulated centroids are off from telemetry by 0.5px
    r, c = np.mgrid[-halfsize:halfsize, -halfsize:halfsize] - 0.5

    # Model
    sigma = FWHM / (2. * np.sqrt(2. * np.log(2.)))
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
    
    # Otherwise mag does not match with telemetry
    g = counts * g / g[r1: r2, c1: c2].sum()

    # Simulate star
    star = ACAImage(np.random.normal(g), row0=row0, col0=col0)

    return star


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

    
def subtract_bgd(img):
    """
    Subtract average background, flight algorithm.
    """
    sz = img.IMGSIZE
    if sz == 6:
        readout = ACAImage(row0=img.row0 - 1, col0=img.col0 - 1, shape=(8, 8))
        bgdimg = img.IMGRAW[readout]
    elif sz == 8:
        bgdimg = img
    else:
        bgdimg = np.zeros((8, 8))

    bgdavg = calc_bgd(bgdimg)
    img = img - bgdavg                    
    img.BGDAVG = bgdavg
        
    return img
    
    
def calc_centroids(img):
    sz = img.IMGSIZE       
    if sz == 6:
        img[[0, 0, 5, 5], [0, 5, 0, 5]] = 0.0  # Mouse-bitten
                
    rw, cw = np.mgrid[0:sz, 0:sz] + 0.5
          
    img_sum = 0.
    col = -511
    row = -511
            
    if img.IMGFUNC1 in (1, 3):
        img_sum = norm = np.sum(img)
        col = np.sum(cw * img) / norm + img.col0
        row = np.sum(rw * img) / norm + img.row0

    return img_sum, row, col
