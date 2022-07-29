
from . import utils
import camelot as cl
import pandas as pd
import numpy as np
from scipy.spatial import ConvexHull
from scipy.interpolate import interp1d


def sample_meta(file):
    """
    Returns a dictionary of the meta data
    from a PDF file (file)
    """
    df = cl.read_pdf(file)[0].df
    meta = {k: v for k, v in zip(df.to_dict('series')[0], df.to_dict('series')[1])}
    return meta


class Spectrum(object):
    """
    Contains everything about a spectroscopy measurement.
    """
    def __init__(self, params=None):
        """
        Generic initializer.
        """
        if params == None:
            params = {}

        for k, v in params.items():
            if k and (v is not None):
                setattr(self, k, v)
        
        empty_header = pd.DataFrame(columns=['id', 'mineral_name', 
                                            'mineral_group', 'sample_number',
                                            'formula', 'locality', 'method'])
        self.data = getattr(self, 'data', {})
        self.header = getattr(self, 'header', empty_header)


    def __repr__(self):
        """
        Non-rich representation.
        ToDo: Make this richer.
        """
        return f"Spectrum('{self.header.items()}')"


    def _repr_html_(self):
        """
        Jupyter Notebook magic repr function.
        """
        html = None
        return html


    @property
    def id(self):
        """
        Property. Simply a shortcut to the ID from the header, or the
        empty string if there isn't one.
        """
        try:
            return self.header[self.header.id == 'id'].value.iloc[0]
        except:
            return ''


    @classmethod
    def from_csv(cls,
                 fname,
                 remap=None,
                 funcs=None,
                 data=True,
                 req=None,
                 alias=None,
                 encoding=None,
                 printfname=False,
                 **kwargs,
                 ):
        """
        Constructor. If you have a csv file saved on disk, this creates a spectrum
        object from it.
        Args:
            fname (str or pathlib.Path): The path of the csv file, or a URL to
                one.
            remap (dict): Optional. A dict of 'old': 'new' csv field names.
            funcs (dict): Optional. A dict of 'fields': function() for
                implementing a transform before loading. Can be a lambda.
            data (bool): Optional. Whether to load the data or only the header.
            encoding (str): Optional. the character encoding used when reading
                the csv file in from disk. Not implemented
            printfname (bool): Optional. prints filename before trying to load
                it, for debugging.
        Returns:
            spectrum. The spectrum object.
        """
        fname = utils.to_filename(fname)

        if printfname:
            print(fname)

        # create spectrum from file
        df = pd.read_csv(fname, sep='\t')
        sample_no = df.columns[-1]
        df.rename({df.columns[-1]: 'reflectance'}, inplace=True, axis=1)
        df.columns = [col.lower() for col in df.columns]
        spectrum = cls({'data': df, 
                        'sample_no': sample_no})

        return spectrum


def convex_hull(s, trim_zeros=True, pad_x=(-500, 5000)):
    """
    Calculates the Convex Hull of a set of points by padding the ends with 
    zeros.
    """
    padded_points = np.pad(np.array(s.data.reflectance), 
                           pad_width=1, mode='constant', constant_values=0)
    padded_wavelength = np.pad(np.array(s.data.wavelength), pad_width=1, mode='edge')
    spec_points_2d = np.array([padded_wavelength, padded_points])
    points = np.array([[p, q] for p, q in spec_points_2d.T])
    hull = ConvexHull(points)
    hull_x, hull_y = points[hull.vertices, 0], points[hull.vertices, 1]
    if trim_zeros:
        hull_x, hull_y = hull_x[0:-1], hull_y[0:-1]
    hull_points = np.sort([hull_x, hull_y], axis=-0)
    return hull_points


def chop_zeros(hull_points):
    out_data = []
    for y, x in hull_points.T:
        if y != 0.0:
            out_data.append([x, y])
    out_data = np.array(out_data).T
    return out_data


def resample_hull_curve(s):
    hull_points = convex_hull(s)
    out_data = chop_zeros(hull_points)
    x, y = out_data[0], out_data[1]
    f = interp1d(x, y, bounds_error=False)
    xnew = s.data.wavelength.values
    ynew = f(xnew)
    return ynew