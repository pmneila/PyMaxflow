# -*- encoding:utf-8 -*-

import numpy as np
import _maxflow
from _maxflow import GraphInt, GraphFloat
from version import __version__, __version_str__, \
        __version_core__, __author__, __author_core__

__doc__ = _maxflow.__doc__

Graph = {int:GraphInt, float:GraphFloat}
