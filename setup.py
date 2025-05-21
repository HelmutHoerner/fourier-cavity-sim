# -*- coding: utf-8 -*-
"""
Created on Wed May 21 14:02:09 2025

@author: Helmut Hoerner
"""

from setuptools import setup

setup(
    name='fo_cavity_sim',
    version='1.0.0',
    py_modules=['fo_cavity_sim'],
    include_package_data=True,
    package_data={
        '': ['cmap_custom.csv'],
    },
)
