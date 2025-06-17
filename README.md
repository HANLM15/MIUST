# Multi-Level Interaction for Emotion Recognition from Unaligned Speech and Text（MIUST）

This project is the official open-source implementation of the paper *“Multi-Level Interaction for Emotion Recognition from Unaligned Speech and Text”*, primarily intended to reproduce and validate the methods and experimental results presented in the paper.

**Authors**: Lingmin Han, Xianhong Chen, Maoshen Jia, and Changchun Bao

------

### 🛠 Requirements

- `pytorch==2.2.1`
- `python==3.8.18`
- `yaml==0.2.5`
- `numpy==1.24.4`
- `scikit-learn==1.3.2`

------

### 📂 Datasets

- [IEMOCAP](https://sail.usc.edu/iemocap/)
- [MELD](https://affective-meld.github.io/)

------

### 📄 Code Structure

#### IEMOCAP Experiments

The `IEMOCAP_exp` directory contains experiments on the IEMOCAP dataset, where 5,531 utterances are used for 4-class emotion classification (anger, happiness + excited, neutral, sadness). A 5-fold cross-validation strategy is used for training.

- `bert_iemocap/`: Pre-extracted BERT text features from the IEMOCAP dataset.
- `emotion2vec_iemocap/`: Pre-extracted emotion2vec audio features from the IEMOCAP dataset.
- `iemocap_csv/`: CSV files corresponding to each session in the IEMOCAP dataset.
- `config/config.yml`: Configuration file for the experiments.
- **`main.py`**: Main program for running experiments.
- `utils.py`: Various utility methods used throughout.
- `dataloader.py`: Construction process for training, validation, and testing data loaders.
- `model.py`: File containing the model definition.
- `output/`: Directory for storing experimental result logs.



#### MELD Experiments

The `MELD_exp` directory contains experiments on the MELD dataset, which uses 13,705 utterances for 7-class emotion classification (anger, disgust, fear, joy, neutral, sadness, surprise). The dataset is split into: 9,988 utterances for training, 1,108 utterances for validation and 2,609 utterances for testing.

- `bert_meld/`: Pre-extracted BERT text features from the MELD dataset.
- `emotion2vec_meld/`: Pre-extracted emotion2vec audio features from the MELD dataset.
- `meld_csv/`: CSV files for training, validation, and testing splits in the MELD dataset.

Other files and directory explanations are the same as for IEMOCAP.

