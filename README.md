# Tweaking Features of Ensembles of Machine-Learned Trees
Source code associated with the KDD 2017 research paper entitled "_Interpretable Predictions of Tree-based Ensembles via Actionable Feature Tweaking_" \[paper is available at: [arXiv.org](https://arxiv.org/abs/1706.06691)\]

This repo is made up of **4** scripts which are supposed to be run in the same order as follows:
1.  <code>**dump_paths.py**</code>
2.  <code>**tweak_features.py**</code>
3.  <code>**compute_tweaking_costs.py**</code>
4.  <code>**dump_recommendations.py**</code>

## 1. <code>**dump_paths.py**</code>
The first stage of the pipeline is accomplished by this script. This can be invoked as follows:

<code>**> ./dump_paths.py ${PATH_TO_SERIALIZED_MODEL} ${PATH_TO_OUTPUT_FILE}**</code>

where 

<code>**${PATH_TO_SERIALIZED_MODEL}**</code> is the path to the (binary) file containing a serialized, trained <code>**scikit-learn**</code> estimator (i.e., model).
<code>**${PATH_TO_OUTPUT_FILE}**</code>

## 2. <code>**tweak_features.py**</code>

## 3. <code>**compute_tweaking_costs.py**</code>

## 4. <code>**dump_recommendations.py**</code>

