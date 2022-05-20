Image Caption Generator using an LSTM ANN
==============================

This project was orignally completed as a jupyter notebook assignment for a Deep Learning course at the Technical University of Eindhoven, and is based on the paper [Show and Tell: A Neural Image Caption Generator](https://arxiv.org/abs/1411.4555) by Vinyals et al. in 2015.

Set Up the Project
------------
1. Clone repository and navigate to the base directory. 

2. Setup a new conda environment and make `./data/raw/`, `./data/processed/` and `./models/` directories.
    - ```$ make setup```

3. Activate environment.
   - ```$ conda activate image_caption_generator```

4. (Optional) Run environment test.
    - ```$ make test_environment```

5. Download the data and pretrained model for this project from Google Drive and place in base directory. 
    - [Download Folder](https://drive.google.com/drive/folders/1s2X-gJgibEo6AVff9HqgqJ_1EkIkFrua?usp=sharing)


The Dataset
------------
The raw data in ```data/raw``` contains 8,000 *preprocessed* images and captions per image from the [Flicker8k](https://www.kaggle.com/adityajn105/flickr8k/activity) dataset. The Kaggle data has already been preprocessed in the following ways (code not provided):
- RGB images are rescaled to 128 x 128 x 3
- Captions do not have punctuation or special tokens and are in lower case
- Each caption is now a list of strings e.g. ['the', 'cow', 'jumped', 'over', 'the',' moon']
- Words occuring less than 5 times in the whole corpus have been removed

**Example Image and Caption**

![Example image and caption](https://github.com/elizastarr/image_caption_generator/blob/master/reports/figures/example_train_image.png?raw=true)


Further Preprocessing
------------
1. Obtain 20480-dimensional representations of the images from the first convolutional layer of MobileNetV2 (pretrained on ImageNet).
2. Insert the stop word character '_' at the end of each string. Map the words to integers sorted by frequency using a dictionary.
3. Train-test-validation splits.

To preprocess the data again (not neccessary), run 
- ```$ python src/data/make_dataset.py data/raw data/processed```

Training a Long-Short-Term-Memory Learner
------------
**Purpose:** Learn weights for the caption generating model 

**Inputs:**
1. Image representations
2. Captions (encoded as integers)

**Architecture:**
1. Dense layer: reduce 20480D image representations to 512D image embeddings
2. Embedding layer: map the caption integers to 512D dense caption vectors
3. Concatenation: Concatenate the image and caption embeddings --> (1, 512)+(n, 512)=(1+n, 512)
4. [LSTM layer (Recurrent NN)](https://www.bioinf.jku.at/publications/older/2604.pdf) 
   - LSTM dropout of 0.5
   - Recurrent dropout of 0.1
5. Dense layer with softmax activation

**Output:**
1. Categorical distribution over the words in the corpus

**Training settings:**
- Adam optimizer with learning rate 1e-3 and early stopping using the validation set
- Batch size 100
- Max epochs 100
- Cross-entropy loss
- Report Accuracy

See ```logs/train/``` and ```logs/validation/``` for TensorBoard event files.

To train and recieve final evaluation scores, run:
- ```$ python src/models/train.py```

To load the trained model and recieve final evaluation scores, run:
- ```$ python src/models/train.py --load```

**Final Evaluation on Whole Datasets** 
- Train: Categorical Cross Entropy: 2.25, Categorical Accuracy: 0.72
- Validation: Categorical Cross Entropy: 2.34, Categorical Accuracy: 0.72

Predicting Captions with an LSTM Decoder
------------
We use another LSTM model with the trained weights from the LSTM Learner to predict captions given image representations. 

**Input:**
1. Image representations

**Output:**
1. Caption predictions (encoded as integers) each of length 36

To predict from the decoder and see examples, run:
- ```$ python src/models/predict.py```

To load the predictions and see examples, run:
- ```$ python src/models/predict.py --load```

**10 Sample images and predicted captions**

![10 Sample images and predicted captions](https://github.com/elizastarr/image_caption_generator/blob/master/reports/figures/predictions.png?raw=true)


Caption Analysis
------------
We [analyze](https://github.com/elizastarr/image_caption_generator/blob/master/reports/prediction_analysis.pdf) the captions using BLEU scores in `notebooks/prediction_analysis.ipynb`. BLEU scores range from 0 to 1 (highest) are "a method of automatic machine translation evaluation that is quick, inexpensive, and language-independent, that correlates highly with human evaluation" [(Papineni et al., 2002)](https://aclanthology.org/P02-1040.pdf). 

The histogram below shows the distribution of scores given different n-grams. The independent BLEU-1 scores (using 1-grams) have a mean of 0.72 and maximum of 0.97. The model is slightly better at replicating certain key words than at replicating the word order or set of 2-4 words in a row. 

**10 Sample images and predicted captions**

![Independent BLUE score histogram](https://github.com/elizastarr/image_caption_generator/blob/master/reports/figures/independent_bleu.png?raw=true)

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
<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
