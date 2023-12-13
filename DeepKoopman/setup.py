# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 16:24:15 2022

@author: Ideal
"""

import setuptools 
 
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
 
setuptools.setup(
    name="deepKoopman", 
    version="1.0.0",   
    author="Ideal",   
    author_email="wangmx@stu.jiangnan.edu.cn",   
    description="Deep Koopman-based Modeling and Control",
    long_description=long_description,    
    long_description_content_type="text/markdown",
    url="https://github.com/IdealDD11/DeepKoopman",    
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',   
)
