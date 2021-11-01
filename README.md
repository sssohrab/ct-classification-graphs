# Hierarchical text classification using geometric deep learning

A PyTorch implementation and the required material to accompany our EM-NLP 2021 paper
 ["Classification of hierarchical text using geometric deep learning: the case of clinical trials corpus"](https://arxiv.org/abs/2110.15710).

Our use case is the clinical trials data, which are hierarchically organized text. Along with text embedding using
pre-trained transformers, the idea is to use graph deep learning to furthermore capture the hierarchical structure.

## This repository

### Downloading the data:

- Download the raw clinical trials data, as well as labels in train/valid/test splits from 
[this link](https://zenodo.org/record/5482332#.YVrFwaCxU6U). The raw data is simply downloaded from 
[CTGov website](https://ClinicalTrials.gov/AllAPIJSON.zip), and the labeling procedure is explained in the paper.

- Flatten the content of `AllAPIJSON.zip` into a directory named `<data_dir>/ctgov/raw/` (e.g., using the Linux command 
`$ find -name *.json -exec mv --target raw {} \+`).
- Copy the downloaded splits with the labels into `<data_dir>/ctgov/splits/`.

### Pre-processing:

The raw data downloaded needs processing, firstly to remove label-sensitive content, and secondly to organize the
hierarchical text to a structure suitable for classification. We consider *flat* structures (i.e., multi-fields with 
fixed-order and hence fixed vectorial order when featurized), as opposed to *graph* structures, which is advocated in 
this paper (i.e., an arbitrary graph where the leaf nodes contain free text.)


#### Flat-classification (no graphs):
For the "flat" strctures, 9 important fields (*"SponsorCollaboratorsModule", "OversightModule",
"DescriptionModule", "ConditionsModule", "DesignModule", "ArmsInterventionsModule",
"OutcomesModule", "EligibilityModule", "ContactsLocationsModule"*) are extracted and organized under 
``<data_dir>/ctgov/preproc/flat/`` as json files. For this, run the following command in a UNIX-like terminal:

```
$ cd <repo_dir>/scripts/preprocess
$ python3 preprocess.py
```

Next step is to featurize the free text into fixed-length vectors. This is done either using classical Bag-of-Words and
TF-IDF, or using language models.

For BOW-based representation, as described in the paper, a sparse random projections stage is used to decrease the 
dimensionality of the tokens (from around 1 Million to 1 Thousand). To first train the BOW-object and pickle it under
`<repo_dir>/storage/`:

```
$ python3 fit_bow_obj.py -c config_train_BOW.json
```


For the lm-based feature extraction, pre-trained 
transformers from the Huggingfaces library are used. The model paprameters will automatically be saved in your disk 
when run for the first time. 

To avoid feature extraction during training and at each epoch, we can extract features once and save them to disk 
(e.g., using `h5` format). This is done using:


```
$ cd <repo_dir>/scripts/preprocess
$ python3 digitize_text.py -c config_train.json
```

where the type of feature extraction (BOW or LM), as well as the details are specified in a `config_train.json` file.
Typical exemplar values are provided under `config_train_BOW.json` and `config_train_LM.json`.


Once the vectorial representations are stored to disk, training is done using:

```
cd <repo_dir>/scripts/flat
python3 train -c config_train.json
```

After the training is concluded, the tensorboard run under `<repo_dir>/scripts/flat/runs/` can be used in the inference
time to classify CT's of the test set using:

```
$ pytohn3 eval.py -c config_eval.json
```

where the tensorboard-assigned tag of the trained model should be copied inside `config_eval.json`.


#### Classification using graphs:

Once the pre-processed CT's are saved to disk (see above), and the pickled BOW object is created from the training set, 
the graph-based classification requires first to save the entire database into disk (as a 
`torch_geometric.data.InMemoryDataset`instance). This is done using:

```
$ cd <repo_dir>/scripts/graph/
$ python3 digitize_text.py -c config_train.json
```

where again the details of featurization for each node of the graphs is specified within `config_train.json`. This can 
take long since each CT can have many nodes (around 100) for which a separate feature needs to be extracted using the 
specified featurizer. Moreover, this also requires big amounts of RAM, as the whole database will be stored in memory
for speed. Otherwise, other types of pytorch_geometric dataset classes should be used.

Once the graphs are created and stored under `<data_root>/preproc/graph/<bow/lm>`, the training and evaluation can be 
done similarly to the flat case using:


```
$ cd <repo_dir>/scripts/graph
$ python3 train -c config_train.json
$ pytohn3 eval.py -c config_eval.json
```

Whether to use ``global`` vs. ``selective`` pooling can be specified in the config files. 


## Citations:

- The paper

```
@article{ferdowsi2021classification,
title="Classification of hierarchical text using geometric deep learning: the case of clinical trials corpus",
author="Ferdowsi, Sohrab and
Borissov, Nikolay and
Knafou, Julien and
Amini, Poorya and
Teodoro, Douglas",
booktitle="Proceedings of the 2011 Conference on Empirical Methods in Natural Language Processing",
year="2021"
}

```

- The database:
  
```@dataset{ferdowsi_sohrab_2021_5482332,
  author       = {Ferdowsi, Sohrab and
  Borissov, Nikolay and
  Knafou, Julien and
  Amini, Poorya and
  Teodoro, Douglas},
  title        = {{Classification of hierarchical text using
  geometric deep learning: the case of clinical
  trials corpus}},
  month        = sep,
  year         = 2021,
  publisher    = {Zenodo},
  version      = {1.0.0},
  doi          = {10.5281/zenodo.5482332},
  url          = {https://doi.org/10.5281/zenodo.5482332}
  }
```
