# -*- coding: utf-8 -*-

"""
maxflow.fastmin
===============

``fastmin`` provides implementations of the algorithms for
fast energy minimization described in [BOYKOV01]_: the alpha-expansion
and the alpha-beta-swap.

.. [BOYKOV01] *Fast approximate energy minimization via graph cuts.*
   Yuri Boykov, Olga Veksler and Ramin Zabih. TPAMI 2001.

Currently, the functions in this module are restricted to
grids with von Neumann neighborhood.
"""

import sys
import logging
from itertools import count, combinations
import numpy as np
from ._maxflow import aexpansion_grid_step, abswap_grid_step

logger = logging.getLogger(__name__)


def energy_of_grid_labeling(D, V, labels):
    """
    Returns the energy of the labeling of a grid.
    
    For details about ``D``, ``V`` and ``labels``, see the
    documentation of ``aexpansion_grid``.
    
    Returns the energy of the labeling.
    """
    
    num_labels = D.shape[-1]
    ndim = labels.ndim
    
    # Sum of the unary terms.
    unary = np.sum([D[labels==i,i].sum() for i in range(num_labels)])
    
    slice0 = [slice(None)]*ndim
    slice1 = [slice(None)]*ndim
    # Binary terms.
    binary = 0
    for i in range(ndim):
        slice0[i] = slice(1, None)
        slice1[i] = slice(None, -1)
        
        binary += V[labels[slice0],labels[slice1]].sum()
        
        slice0[i] = slice(None)
        slice1[i] = slice(None)
    
    return unary + binary


def abswap_grid(D, V, max_cycles=None, labels=None):
    """
    Minimize an energy function iterating the alpha-beta-swap
    until convergence or until a maximum number of cycles,
    given by ``max_cycles``, is reached.
    
    ``D`` must be a N+1-dimensional array with shape (S1,...,SN,L),
    where L is the number of labels considered. *D[p1,...,pn,lbl]* is the unary
    cost of assigning the label *lbl* to the variable *(p1,...,pn)*.
    
    ``V`` is a two-dimensional array. *V[lbl1,lbl2]* is the binary cost of
    assigning the labels *lbl1* and *lbl2* to a pair of neighbor variables.
    Note that the abswap algorithm, unlike the aexpansion, does not require
    ``V`` to define a metric.
    
    The optional N-dimensional array ``labels`` gives the initial labeling
    for the algorithm. If not given, the function uses a plain initialization
    with all the labels set to 0.
    
    This function return the labeling reached at the end of the algorithm.
    If the user provides the parameter ``labels``, the algorithm will work
    modifying this array in-place.
    """
    num_labels = D.shape[-1]
    
    if labels is None:
        # Avoid using too much memory.
        if num_labels <= 127:
            labels = np.int8(D.argmin(axis=-1))
        else:
            labels = np.int_(D.argmin(axis=-1))
    
    if max_cycles is None:
        rng = count()
    else:
        rng = range(max_cycles)
    
    prev_labels = np.copy(labels)
    better_energy = np.inf
    # Cycles.
    for i in rng:
        logger.info("Cycle {}...".format(i))
        improved = False
        
        # Iterate through the labels.
        for alpha, beta in combinations(range(num_labels), 2):
            energy, _ = abswap_grid_step(alpha, beta, D, V, labels)
            logger.info("Energy of the last cut (α={}, β={}): {:.6g}".format(alpha, beta, energy))
            
            # Compute the energy of the labeling.
            strimproved = ""
            energy = energy_of_grid_labeling(D, V, labels)
            
            # Check if the better energy has been improved.
            if energy < better_energy:
                prev_labels = np.copy(labels)
                better_energy = energy
                improved = True
                strimproved = "(Improved!)"
            else:
                # If the energy has not been improved, discard the changes.
                labels = prev_labels
            
            logger.info("Energy of the labeling: {:.6g} {}".format(energy, strimproved))
        
        # Finish the minimization when convergence is reached.
        if not improved:
            break
    
    return labels


def aexpansion_grid(D, V, max_cycles=None, labels=None):
    """
    Minimize an energy function iterating the alpha-expansion until
    convergence or until a maximum number of cycles,
    given by ``max_cycles``, is reached.
    
    ``D`` must be an N+1-dimensional array with shape (S1,...,SN,L),
    where L is the number of labels considered. *D[p1,...,pn,lbl]* is the unary
    cost of assigning the label *lbl* to the variable *(p1,...,pn)*.
    
    ``V`` is a two-dimensional array. *V[lbl1,lbl2]* is the binary cost of
    assigning the labels *lbl1* and *lbl2* to a pair of neighbor variables.
    Note that the distance defined by ``V`` must be a metric or the aexpansion
    might fail.
    
    The optional N-dimensional array ``labels`` gives the initial labeling
    of the algorithm. If not given, the function uses a plain initialization
    with all the labels set to 0.
    
    This function return the labeling reached at the end of the algorithm.
    If the user provides the parameter ``labels``, the algorithm will work
    modifying this array in-place.
    """
    num_labels = D.shape[-1]
    
    if labels is None:
        # Avoid using too much memory.
        if num_labels <= 127:
            labels = np.int8(D.argmin(axis=-1))
        else:
            labels = np.int_(D.argmin(axis=-1))
    
    if max_cycles is None:
        rng = count()
    else:
        rng = range(max_cycles)
    
    better_energy = np.inf
    # Cycles.
    for i in rng:
        logger.info("Cycle {}...".format(i))
        improved = False
        # Iterate through the labels.
        for alpha in range(num_labels):
            energy, _ = aexpansion_grid_step(alpha, D, V, labels)
            strimproved = ""
            # Check if the better energy has been improved.
            if energy < better_energy:
                better_energy = energy
                improved = True
                strimproved = "(Improved!)"
            logger.info("Energy of the last cut (α={}): {:.6g} {}".format(alpha, energy, strimproved))
        
        # Finish the minimization when convergence is reached.
        if not improved:
            break
    return labels
