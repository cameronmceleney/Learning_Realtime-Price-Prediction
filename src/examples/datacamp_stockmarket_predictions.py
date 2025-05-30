#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Detailed worked example of making stock market predictions using Long Short-Term Memory (LSTM) networks.

This file is based upon an excellent tutorial by `Thushan Ganegedara which can be found on Datacamp`_. This article
provided a great introduction to the topic, and was more complex than the LSTM example given  in
`medium_article_realtime_predictions.py`. In contrast, this file is not a faithful reproduction of the tutorial, but
rather a more complex example that builds upon the concepts introduced in the tutorial.

This file is intended to be used as a reference for me in the future, and to provide a more comprehensive understanding
of LSTM networks in the context of stock market predictions.

Constants:
    MODULE_LEVEL_CONSTANT1 (int): A module-level constant.

Examples:
    (Any example implementations for this file)::
        
        $ bar = 1
        $ foo = bar + 1

Todo:
    * Work through referenced (url) tutorial and implement the code in this file.

References:
    Article author: Thushan Ganegedara

    Article created on: 10 Dec 2024

    Style guide: `Google Python Style Guide`_

Notes:
    File version
        0.1.0
    Project
        Learning_Realtime-Price-Prediction_toolkit
    Path
        src/examples/datacamp_stockmarket_predictions.py
    Author
        Cameron Aidan McEleney < c.mceleney.1@research.gla.ac.uk >
    Created
        29 May 2025
    IDE
        PyCharm
        
.. _Google Python Style Guide:
   https://google.github.io/styleguide/pyguide.html

.. _Thushan Ganegedara which can be found on Datacamp:
    https://www.datacamp.com/tutorial/lstm-python-stock-market
"""

__all__ = [""]

# Standard library imports
from typing import Any

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np

# Local application imports

# Module-level constants
MODULE_LEVEL_CONSTANT1: int = 1
"""A module-level constant with in-line docstring."""


def my_func() -> None:
    """(One-line summary of function.)"""
    # Do stuff
    pass


if __name__ == "__main__":
    my_func()
