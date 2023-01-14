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

import logging
from itertools import count, combinations
import numpy as np
from ._maxflow import aexpansion_grid_step, abswap_grid_step

logger = logging.getLogger(__name__)


def energy_of_grid_labeling(unary, binary, labels):
    """
    Returns the energy of the labeling of a grid.

    For details about ``unary``, ``binary`` and ``labels``, see the
    documentation of ``aexpansion_grid``.

    Returns the energy of the labeling.
    """

    num_labels = unary.shape[-1]
    ndim = labels.ndim

    # Sum of the unary terms.
    unary_energy = np.sum([unary[labels == i, i].sum() for i in range(num_labels)])

    slice0 = [slice(None)]*ndim
    slice1 = [slice(None)]*ndim
    # Binary terms.
    binary_energy = 0
    for i in range(ndim):
        slice0[i] = slice(1, None)
        slice1[i] = slice(None, -1)

        binary_energy += binary[labels[tuple(slice0)], labels[tuple(slice1)]].sum()

        slice0[i] = slice(None)
        slice1[i] = slice(None)

    return unary_energy + binary_energy


def abswap_grid(unary, binary, max_cycles=None, labels=None):
    """
    Minimize an energy function iterating the alpha-beta-swap until convergence
    or until a maximum number of cycles, given by ``max_cycles``, is reached.

    ``unary`` must be a N+1-dimensional array with shape (S1, ..., SN, L), where
    L is the number of labels. *unary[p1, ..., pn, lbl]* is the unary cost of
    assigning the label *lbl* to the variable *(p1, ..., pn)*.

    ``binary`` is a two-dimensional array. *binary[lbl1, lbl2]* is the binary
    cost of assigning the labels *lbl1* and *lbl2* to a pair of neighbor
    variables. Note that the abswap algorithm, unlike the aexpansion, does not
    require ``binary`` to define a metric.

    The optional N-dimensional array ``labels`` gives the initial labeling for
    the algorithm. If omitted, the function will initialize the labels using the
    minimum unary costs given by ``unary``.

    The function return the labeling reached at the end of the algorithm. If the
    parameter ``labels`` is given, the function will modify this array in-place.
    """
    if unary.ndim == 0:
        raise ValueError("The unary term cannot be a scalar")

    num_labels = unary.shape[-1]

    if num_labels == 0:
        raise ValueError("The number of labels cannot be 0")

    if binary.shape != (num_labels, num_labels):
        raise ValueError(
            "The binary term must be a square matrix of shape (num_labels, num_labels). "
            f"The shape of the binary term was {binary.shape} but, according to the unary term, num_labels={num_labels}"
        )

    if labels is None:
        # Avoid using too much memory.
        if num_labels <= 127:
            labels = np.int8(unary.argmin(axis=-1))
        else:
            labels = np.int_(unary.argmin(axis=-1))
    else:
        if labels.min() < 0:
            raise ValueError("Values of labels must be non-negative")
        if labels.max() >= num_labels:
            raise ValueError(f"Values of labels must be smaller than num_labels={num_labels}")

    if max_cycles is None:
        rng = count()
    else:
        rng = range(max_cycles)

    prev_labels = np.copy(labels)
    best_energy = np.inf
    # Cycles.
    for i in rng:
        logger.info("Cycle {}...".format(i))
        improved = False

        # Iterate through the labels.
        for alpha, beta in combinations(range(num_labels), 2):
            energy, _ = abswap_grid_step(alpha, beta, unary, binary, labels)
            logger.info("Energy of the last cut (α={}, β={}): {:.6g}".format(alpha, beta, energy))

            # Compute the energy of the labeling.
            strimproved = ""
            energy = energy_of_grid_labeling(unary, binary, labels)

            # Check if the best energy has been improved.
            if energy < best_energy:
                prev_labels = np.copy(labels)
                best_energy = energy
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


def aexpansion_grid(unary, binary, max_cycles=None, labels=None):
    """
    Minimize an energy function iterating the alpha-expansion until convergence
    or until a maximum number of cycles, given by ``max_cycles``, is reached.

    ``unary`` must be an N+1-dimensional array with shape (S1, ..., SN, L),
    where L is the number of labels. *unary[p1, ... ,pn ,lbl]* is the unary cost
    of assigning the label *lbl* to the variable *(p1, ..., pn)*.

    ``binary`` is a two-dimensional array. *binary[lbl1, lbl2]* is the binary
    cost of assigning the labels *lbl1* and *lbl2* to a pair of neighbor
    variables. Note that the distance defined by ``binary`` must be a metric or
    else the aexpansion might converge to invalid results.

    The optional N-dimensional array ``labels`` gives the initial labeling of
    the algorithm. If omitted, the function will initialize the labels using the
    minimum unary costs given by ``unary``.

    The function return the labeling reached at the end of the algorithm. If the
    parameter ``labels`` is given, the function will modify this array in-place.
    """
    if unary.ndim == 0:
        raise ValueError("The unary term cannot be a scalar")

    num_labels = unary.shape[-1]

    if num_labels == 0:
        raise ValueError("The number of labels cannot be 0")

    if binary.shape != (num_labels, num_labels):
        raise ValueError(
            "The binary term must be a square matrix of shape (num_labels, num_labels). "
            f"The shape of the binary term was {binary.shape} but, according to the unary term, num_labels={num_labels}"
        )

    if labels is None:
        # Avoid using too much memory.
        if num_labels <= 127:
            labels = np.int8(unary.argmin(axis=-1))
        else:
            labels = np.int_(unary.argmin(axis=-1))
    else:
        if labels.min() < 0:
            raise ValueError("Values of labels must be non-negative")
        if labels.max() >= num_labels:
            raise ValueError(f"Values of labels must be smaller than num_labels={num_labels}")

    if max_cycles is None:
        rng = count()
    else:
        rng = range(max_cycles)

    best_energy = np.inf
    # Cycles.
    for i in rng:
        logger.info("Cycle {}...".format(i))
        improved = False
        # Iterate through the labels.
        for alpha in range(num_labels):
            energy, _ = aexpansion_grid_step(alpha, unary, binary, labels)
            strimproved = ""
            # Check if the best energy has been improved.
            if energy < best_energy:
                best_energy = energy
                improved = True
                strimproved = "(Improved!)"
            logger.info("Energy of the last cut (α={}): {:.6g} {}".format(alpha, energy, strimproved))

        # Finish the minimization when convergence is reached.
        if not improved:
            break

    return labels
