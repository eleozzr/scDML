# scDML

## overview

scDML (Batch Alignment of **s**ingle **c**ell transcriptomics data using **D**eep **M**etric **L**earning) is designed for single cell transcriptomics data's clustering ,which is a novel method based on deep metric learning to remove batch effect, it is implemented with pytorch and utilizes the [pytroch_metric_learning](https://kevinmusgrave.github.io/pytorch-metric-learning/) package .  Workflow of scDML is below:

![](./images/workflow1.png)

you can get more detail to see our manualscript.

## merge rule

scDML adopt  a novel merge rule to reassign cluster labels, which is crucial step in scDML method. you can this is the semantic dynamic display of merge rule ,which is the 

![](./images/init_cluster.png)



![](./images/scDML_merge_rule.gif)

## gettting started

see [tutorial1](./tutorial/tutorial1.ipynb) and  [tutorial2](./tutorial/tutorial2.ipynb) , where **tutorial1** gives a detailed discription in each step of scDML, while **tutorial 2** provides a complete running example 

## installation(test on linux)

recomended installation procedure is as follows.

### step1 

1. Install [Anaconda](https://www.anaconda.com/products/individual) if you do not already have it.
2. Create a conda environment with python, and then activate it as follows in terminal.

```python
conda create -n scDML python==3.6.10
conda activate scDML
```

### step2

you can install necessary requirments

```python
pip install -r requirements.txt -i https://pypi.douban.com/simple
cd code
python setup.py install --user
```

or 

```python
####### pip install scDML(to be released)   #################
```

### step3

create kernel to run the tutorial

```python
conda activate scDML
python -m ipykernel install --name scDML
```

then you can run the tutorial.ipynb

## Software Requirements

```python
numba==0.51.2
numexpr==2.7.1
numpy==1.18.1 
numpy-groupies==0.9.14
anndata==0.7.6
tables==3.6.1
scanpy==1.7.2
umap-learn==0.4.6
python-igraph==0.8.2
louvain==0.7.0
plotly==5.2.2
hnswlib==0.5.2
annoy==1.17.0
networkx==2.5
ipykernel==5.5.6
nbformat==5.1.3
pytorch-metric-learning==0.9.95
torch>=1.10.1
```

## Citation

