# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 09:42:57 2024

@author: PKUN
"""

import numpy as np



# optional
def l1_normalize(values: np.ndarray) -> np.ndarray:
    """
    Normalizes a vector to have a max of 1.

    Parameters
    ----------
    values : np.ndarray
        Array of data.

    Returns
    -------
    Numpy.ndarray
        Normalized vector with a maximum of 1.

    """
    # retrieve the highest value from the array
    highest_value = np.max(values)
    # if the highest is 0, just return the value, otherwise norm it
    return values if np.isclose(highest_value, 0) else values/highest_value
    
# in use
def surface_normalize(values:np.ndarray) -> np.ndarray:
    """
    Normalises the array to have a sum of 1.

    Parameters
    ----------
    values : np.ndarray
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """    
    # retrieve the sum of the values
    total = np.sum(values)
    # if the total is 0, return the values, otherwise norm it
    return values if np.isclose(total, 0) else values / total
    


def normalized_density(requested_values: np.ndarray, 
                       base_density, 
                       normalization=surface_normalize):
    return normalization(base_density(requested_values))


def test_l1_normalize():
    values_a = np.array([1.,2.])
    normed_a = l1_normalize(values_a)
    
    top_value = max(normed_a)
    
    assert np.isclose(top_value, 1.0), "Top value of l1 norm is not equal to 1.0"
    
    
    


def test_surface_normalize():
    # The smallest values are now
    values_a = np.array([0.,1.])
    values_b = np.array([1.,1.])
    
    normed_a = surface_normalize(values_a)
    normed_b = surface_normalize(values_b)
    
    total = sum(min(normed_a[i], normed_b[i]) for i in range(len(values_a)))
    
    assert np.isclose(total, 0.5), "Surface normalization did not yield the correct result."
    
    
    
if __name__ == "__main__":
    test_surface_normalize()
    test_l1_normalize()

