# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 12:22:49 2024

@author: PKUN
"""
import pandas as pd
import numpy as np
from typing import Callable
from scipy.stats import gaussian_kde

density_function = Callable[[np.ndarray], np.ndarray]

def compute_density_function(data:pd.Series, kernel_width:float|None=None) -> density_function:
    """
    Computes the density function for a given series.
    It is a wrapper with the option to standardise the kernel_width.
    Please read the docs at scipy.stats.gaussian_kde. 

    Parameters
    ----------
    data : pd.Series
        Series of floating point numbers (and potentially NaN values).
    kernel_width : float|None, optional
        The covariance factor. The default is None.

    Returns
    -------
    density_function
        Callable which, for a given set of values, will compute the density at these values.

    """
    # compute the density after removing all NaN values from the series
    density = gaussian_kde(data.dropna())
    # if a value is given for Kernelwith
    if kernel_width:
        density.covariance_factor = lambda: kernel_width
        density._compute_covariance()
    return density
    
    

def density_overlap(density_a: density_function, 
                    density_b: density_function, 
                    start_range:float = -1.0,
                    end_range:float= 1.0,
                    resolution:int=1000) -> float:
    """
    Computes the area under the given density functions.

    Parameters
    ----------
    density_a : density_function
        Density function given by the compute_density_function.
    density_b : density_function
        Density function given by the compute_density_function
    start_range : float, optional
        Where to start (ending included). The default is 0.0.
    end_range: float, optional
        Whete to end (included in the result). The default is 1.0
    resolution : int, optional
        How many steps to consider between the points. The default is 1000.

    Returns
    -------
    float
        surface overlap between the densities.

    """
    # the resolution is the number of steps to investigate, but our method requires the sensitivity which describes the size per step
    sensitivity = 1/resolution
    # we are interested in the full range of values, which starts at the lowest, but should include the top
    scan_values = np.arange(start_range, end_range+sensitivity, 
                            sensitivity)
    # for each value of the range, we now make a 1 on 1 value overview
    densities = zip(density_a(scan_values), density_b(scan_values))
    # we are interested in the common overlap, meaning the lowest of either for
    # any given value of X. 
    return sum(map(min,densities))
    

def density_plot_data(densities:dict[str,density_function], 
                 start_range:float = -1.0,
                 end_range:float= 1.0,
                 resolution:int=1000,
                 ) -> pd.DataFrame:
    # the resolution is the number of steps to investigate, but our method requires the sensitivity which describes the size per step
    sensitivity = 1/resolution
    # we are interested in the full range of values, which starts at the lowest, but should include the top
    scan_values = np.arange(start_range, end_range+sensitivity, 
                            sensitivity)
    density_frame = pd.concat(
        [pd.Series(data=density_f(scan_values), name=density_name)
         for density_name, density_f in iter(densities.items())]
        ,axis=1)
    density_frame['cosine similarity'] = scan_values
    density_frame.set_index('cosine similarity', inplace=True)
    return density_frame

def plot_density(densities:dict[str,density_function], 
                 start_range:float = -1.0,
                 end_range:float= 1.0,
                 resolution:int=1000,
                 **kwargs):
    density_frame = density_plot_data(densities, start_range, end_range, resolution)
    density_frame.plot.line(**kwargs)
    

    
    
def store_plot_density(densities:dict[str,density_function], 
                       file_path:str,
                       start_range:float = -1.0,
                       end_range:float= 1.0,
                       resolution:int=1000,
                       **kwargs):
    # the resolution is the number of steps to investigate, but our method requires the sensitivity which describes the size per step
    sensitivity = 1/resolution
    # we are interested in the full range of values, which starts at the lowest, but should include the top
    scan_values = np.arange(start_range, end_range+sensitivity, 
                            sensitivity)
    density_frame = pd.concat(
        [pd.Series(data=density_f(scan_values), name=density_name)
         for density_name, density_f in iter(densities.items())]
        ,axis=1)
    density_frame['cosine similarity'] = scan_values
    density_frame.set_index('cosine similarity', inplace=True)
    pl = density_frame.plot.line(**kwargs)
    pl.figure.savefig(file_path)
    
    
    
class histogram_density(object):        
    
    def __init__(self, 
                 resolution:int, 
                 minimal_value:float, 
                 maximum_value:float,
                 precision: int=2):
        """
        :param resolution: number of steps to include, this excludes the +1 for the included maximum
        :param maximum_value: included maximum in the range
        """
        self._precision = precision
        self._step_size = (maximum_value - minimal_value) / resolution
        indexed_values = (minimal_value + (self._step_size * i) for i in range(resolution+1))        
        self._recorded = {np.round(value, precision):0 for value in indexed_values}
        
                
    def __add__(self, value:float):
        self._recorded[np.round(value, self._precision)] += 1
        
    def histogram(self)->dict[float, int]:
        return self._recorded.copy()
        
    