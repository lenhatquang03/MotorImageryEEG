# EEG-based human identification with Machine learning and Deep learning
<details>
  <summary><strong>Table of Contents</strong></summary>
  
  1. [About the Project](#about-the-project)
      - [Built With](#built-with)
  2. [Getting Started](#getting-started)
      - [Installation](#installation)
  3. [Usage](#usage)
      - [Data Crawling](#data-crawling)
      - [Project Structure](#project-structure)
     
</details>

## About The Project
This is my work for the Project II module at HUST. In this project, **Fourier transform**, **Wavelet transform**, and **windowing** are applied to extract features from the raw EEG data of 7 individuals. These features were then fit by advanced techniques such as **Support Vector Machine (SVM)**, **Convolutional Neural Network (CNN)**, and **Self-Attention Transformer** for classification.

### Built With
- ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
- ![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
- ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
- ![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
- ![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
- ![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)
- ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=matplotlib&logoColor=white)
- ![MNE-Python](https://img.shields.io/badge/MNE-Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
- ![Joblib](https://img.shields.io/badge/Joblib-005C9C?style=for-the-badge&logo=python&logoColor=white)
- ![Scrapy](https://img.shields.io/badge/Scrapy-9A1E23?style=for-the-badge&logo=python&logoColor=white)
- ![Selenium](https://img.shields.io/badge/Selenium-43B02A?style=for-the-badge&logo=selenium&logoColor=white)

## Getting Started
### Installation
1. Clone the repository
   ```
   git clone https://github.com/lenhatquang03/MotorImageryEEG.git
   ```
2. Navigate to the porject directory
   ```
   cd MotorImageryEEG
   ```
2. Setup environment
   ```
   pip install -r requirements.txt
   ```

## Usage
### Data Crawling
The data used here is a part of a larger EEG motor imagery dataset with different interaction paradigm. Those of **the NoMT (No motor imagery) paradigm** were purposefully chosen for this project. More information can be found here: https://www.nature.com/articles/sdata2018211#Sec23
<br> ⚠️ Please change the <code>download_dir</code> class variable of <code>FigshareSpider</code> in the file <code>figshare_spider.py</code>. 
<br> ⚠️ It is recommended that the data is saved in an external hard drive and only relevant data files are moved into the project directory.
<br> Navigate to the Scrapy project and run the spider:
   ```
   cd EEG_BCI
   scrapy crawl figshare
   ```
### Project Structure
- <code>classifiers</code>: Include checkpoints of the CNN model in cross-validation, trained models, and Jupyter notebooks
- <code>data</code>: EEG .mat data files for the 7 participants and Wavelet-extracted features
- <code>EEG_BCI</code>: The Scrapy project for data crawling
- <code>DataFile.py</code>: A class to extract relevent data fields from .mat files
- <code>FeatureExtractor.py</code>: A class to extract features in frequency and time-frequency domains
- <code>Utilities.py</code>: Contain utility functions
