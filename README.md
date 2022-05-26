Image Caption Generator using an LSTM ANN
==============================

This project was orignally completed as a jupyter notebook assignment for a Deep Learning course at the Technical University of Eindhoven, and is based on the paper [Show and Tell: A Neural Image Caption Generator](https://arxiv.org/abs/1411.4555) by Vinyals et al. in 2015.

Set Up the Project
------------
1. Clone repository and navigate to the base directory.

2. Setup a new conda environment.
    - ```$ make setup_environment```

3. Activate environment.
   - ```$ conda activate image_caption_generator```

5. Download the data and pretrained model for this project from Google Drive and place in base directory.
    - [Download Folder](https://drive.google.com/drive/folders/1s2X-gJgibEo6AVff9HqgqJ_1EkIkFrua?usp=sharing)


The Dataset
------------
The raw data in ```data/raw``` contains 8,000 *preprocessed* images and 5 captions per image from the [Flicker8k](https://www.kaggle.com/adityajn105/flickr8k/activity) dataset. The Kaggle data has already been preprocessed in the following ways (code not provided):
- RGB images are rescaled to 128 x 128 x 3
- Captions do not have punctuation or special tokens and are in lower case
- Each caption is now a list of strings e.g. ['the', 'cow', 'jumped', 'over', 'the',' moon']
- Words occuring less than 5 times in the whole corpus have been removed

**Example Image and Caption**

![Example image and caption](https://github.com/elizastarr/image_caption_generator/blob/master/reports/figures/example_train_image.png?raw=true)


Further Preprocessing
------------
The additional preprocessing steps listed below are required to obtain the training, validation, and test datasets.
1. Obtain 20480-dimensional representations of the images from the first convolutional layer of MobileNetV2 (pretrained on ImageNet).
2. Insert the stop word character '_' at the end of each caption. Map the words to integers sorted by frequency using a dictionary.
3. Train-test-validation splits.

To preprocess the data in `data/raw` and save the data in `data/processed`, run
- ```$ python src/make_dataset.py```

Training a Long-Short-Term-Memory Learner
------------
**Purpose:** Learn weights for the caption generating model (the decoder)

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
- ```$ python src/train.py```

**Final Evaluation on Whole Datasets**
- Train: Categorical Cross Entropy: 2.25, Categorical Accuracy: 0.72
- Validation: Categorical Cross Entropy: 2.34, Categorical Accuracy: 0.72

Predicting Captions with an LSTM Decoder
------------
We use another LSTM with the trained weights from the LSTM Learner to predict captions given image representations.

**Input:**
1. Image representations

**Output:**
1. Caption predictions (encoded as integers) each of length 36

To predict from the decoder and see examples, run:
- ```$ python src/predict.py```

To load the predictions and see examples, run:
- ```$ python src/predict.py --load```

**10 Sample images and predicted captions**

![10 Sample images and predicted captions](https://github.com/elizastarr/image_caption_generator/blob/master/reports/figures/predictions.png?raw=true)


Caption Analysis
------------
We [analyze](https://github.com/elizastarr/image_caption_generator/blob/master/reports/prediction_analysis.pdf) the captions using BLEU scores in `notebooks/prediction_analysis.ipynb`. BLEU scores range from 0 to 1 (highest) are "a method of automatic machine translation evaluation that is quick, inexpensive, and language-independent, that correlates highly with human evaluation" [(Papineni et al., 2002)](https://aclanthology.org/P02-1040.pdf).

The histogram below shows the distribution of scores given different n-grams. The independent BLEU-1 scores (using 1-grams) have a mean of 0.72 and maximum of 0.97. The model is slightly better at replicating certain key words than at replicating the word order or set of 2-4 words in a row.

**10 Sample images and predicted captions**

![Independent BLUE score histogram](https://github.com/elizastarr/image_caption_generator/blob/master/reports/figures/independent_bleu.png?raw=true)

Tests
------------
To run all code tests:

```python -m pytest```

or

```pytest tests```

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile to quickly set up the project.
    ├── README.md          <- Instructions for the project.
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported.
    ├── tox.ini            <- tox file with settings for running tox.
    ├── environment.yaml   <- The conda requirements installed with `make setup_environment`
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries.
    │
    ├── notebooks          <- Jupyter notebooks.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting.
    │
    ├── config             <- Primary configurations and Config class used throughout the project
    │
    ├── tests              <- Pytests
    │
    ├── data
    │   ├── processed      <- The final data sets for model training, validation and inference.
    │   └── raw            <- The original, immutable data dump.
    │
    └── src                <- Source code for use in this project. Scripts are located at the base of this folder.
        │
        ├── predict.py     <- Script to run inference on the trained model.
        ├── train.py       <- Script to rain the model.
        ├── make_dataset.py<- Script to preprocesses the dataset.
        ├── visualize_dataset.py
        |
        ├── data_utils     <- Useful functions for data preprocessing.
        │   │
        │   ├── caption_preprocessing.py
        │   ├── image_representations.py
        │   ├── split_and_format.py
        │   └── load_data.py
        │
        ├── models         <- Model class definitions.
        │   │
        │   ├── decoder.py
        │   └── LSTMLearner.py
        │
        └── analysis_utils <- Useful functions for evaluating the model and exploring the data.
            │
            ├── bleu_scores.py
            └── visualization.py


Acknowledgements
------------
<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
