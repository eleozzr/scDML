#!/usr/bin/env python
# coding: utf-8

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="scDML",
    version="0.0.1",
    author="xiaokangyu",
    author_email="yuxiaokang2018@163.com",
    description="Batch Alignment of single-cell transcriptomics data using Deep Metric Learning.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/eleozzr/scDML",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=['numba>=0.51.2',"numexpr>=2.7.1","numpy-groupies>=0.9.14",'numpy>=1.18.1',
    'anndata>=0.7.6','tables>=3.6.1','scanpy>=1.7.2',"umap-learn>=0.4.6","python-igraph>=0.8.2",
     'louvain>=0.7.0',"plotly>=5.2.2","hnswlib>=0.5.2","annoy>=1.17.0","networkx>=2.5",
     'torch>= 1.6.0',"ipykernel>=5.5.6","nbformat>=5.1.3","pytorch-metric-learning>=0.9.95"],
    python_requires='>=3.6',
)
