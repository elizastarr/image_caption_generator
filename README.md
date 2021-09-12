Image Caption Generator using an LSTM ANN
==============================

Code for the second technical interview at PTTRNS.ai.


The Dataset
------------
The raw data in ```data/raw``` and ```data/processed``` contains 8,000 preprocessed images and captions per image from the [Flicker8k](https://www.kaggle.com/adityajn105/flickr8k/activity) dataset "for sentence-based image description and search, consisting of 8,000 images that are each paired with five different captions which provide clear descriptions of the salient entities and events." The Kaggle data has already been preprocessed in the following ways:
- RGB images are rescaled to 128 x 128 x 3
- Captions do not have punctuation or special tokens and are in lower case
- Each caption is now a list of strings e.g. ['the', 'cow', 'jumped', 'over', 'the',' moon']
- Words occuring less than 5 times in the whole corpus have been removed

Preprocessing
------------
1. Obtain 20480-dimensional representations of the images from the first convolutional layer of MobileNetV2 (pretrained on ImageNet).
2. Insert the stop word character '_' at the end of each string. Map the words to integers sorted by frequency using a dictionary.
3. Train-test-validation splits.

To Run:
```
$ python src/data/make_dataset.py data/raw data/processed
```

The Long-Short-Term-Memory Model 
------------
Purpose:
Learn weights for the caption generating model 

Inputs:
1. Image representations
2. Captions (encoded as integers)

Architecture:
1. Dense layer: reduce 20480D image representations to 512D image embeddings
2. Embedding layer: map the caption integers to 512D dense caption vectors (["the position of a word in the vector space is learned from text and is based on the words that surround the word when it is used"](https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/))
3. Concatenation: Concatenate the image and caption embeddings --> (1, 512)+(n, 512)=(1+n, 512)
4. [LSTM layer (Recurrent NN)](https://www.bioinf.jku.at/publications/older/2604.pdf) 
   - LSTM dropout of 0.5
   - Recurrent dropout of 0.1
5. Dense layer with softmax activation

Output:
1. Categorical distribution over the words in the corpus

Total parameters: 15,543,168

Training settings:
- Adam optimizer with learning rate 1e-3 and early stopping using the validation set
- Batch size 100
- Max epochs 100
- Cross-entropy loss
- Report Accuracy

![alt text](https://github.com/elizastarr/[reponame]/blob/[branch]/image.jpg?raw=true)
![alt text](https://github.com/elizastarr/[reponame]/blob/[branch]/image.jpg?raw=true)

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to preprocess data
    │   │   │                
    │   │   ├── caption_preprocessing.py
    │   │   ├── image_representations.py
    │   │   └── make_dataset.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Modules and scripts for exploratory and results-oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


Acknowledgements
------------

This project was orignally completed as an assignment for a Deep Learning course at the Technical University of Eindhoven, and is based on the paper [Show and Tell: A Neural Image Caption Generator](https://arxiv.org/abs/1411.4555) by Vinyals et al. in 2015.

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>