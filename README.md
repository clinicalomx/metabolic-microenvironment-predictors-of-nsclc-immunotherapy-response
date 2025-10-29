# metabolic-microenvironment-predictors-of-nsclc-immunotherapy-response
Code for manuscript "Metabolic characterization of tumor-immune interactions by multiplexed immunofluorescence reveals spatial mechanisms of immunotherapy response in non-small cell lung carcinoma (NSCLC)"

univariate_analysis contains code for figures 1-3

multivariate_analysis contains notebooks for figures 4-7

feature_generation contains scripts called in multivairate_analysis notebooks, including neighbourhood and metabolic neighbourhood generation,and spatial feature generation 

Base conda environment setup:
  
  conda create -n analysis python=3.10 matplotlib seaborn jupyterlab anndata scanpy
  conda activate analysis
  pip install squidpy
  
  pip install PyComplexHeatmap glasbey scikit-survival
  
  pip install git+https://github.com/gregbellan/Stabl.git@v1.0.1-lw
  pip install scikit-learn==1.5.2
