# Speaker Recognition using SideKit
This repo contains my Speaker Recognition/Verification project using SideKit.


## SideKit
SIDEKIT is an open source package for Speaker and Language recognition. The aim of SIDEKIT is to provide an educational and efficient toolkit for speaker/language recognition including the whole chain of treatment that goes from the audio data to the analysis of the system performance.

Authors:	Anthony Larcher & Kong Aik Lee & Sylvain Meignier
Version:	1.3.1 of 2019/01/22
You can check the official documentation, altough I don't recommend it, from [here](https://projets-lium.univ-lemans.fr/sidekit/). Also [here](https://projets-lium.univ-lemans.fr/sidekit/api/index.html) is the API documentation.


To run SIDEKIT on your machine, you need to:

- Install the dependencies by running `pip install -r requirements.txt`
- Install `pytorch` by running the following command: `pip3 install https://download.pytorch.org/whl/cpu/torch-1.0.1.post2-cp36-cp36m-linux_x86_64.whl`.
- Install `torchvision` by running `pip3 install torchvision`.
- Install tkinter by running `sudo apt-get install python3.6-tk`.
- Install libSVM by running `sudo apt-get install libsvm-dev`. This library dedicated to SVM classifiers.


As we can see, the pipeline consists of six main steps:

- **Structure**: In this step, we produce some files that will be helpful when training and evaluating our model.
- **Feature Extraction**: In this step, we extract pre-defind features from the wav files.
- **Choosing A Model**: In this step, we choose a certain model, out of four, to be trained. We have five models that can be trained:
	- UBM
	- SVM with GMM (NOT READY YET)
	- I-vector
	- Deep Learning (NOT READY YET)
- **Training**: This step is pretty self-explanatory ... com'on.
- **Evaluating**: This step is used to evaluate our model using a test set.

All the configuration options for all previous steps can be found in `config.ini`.


### 1. Preprocessing

The file responsible for data structure in `data_init.py` should be following format: 

- sample rate to 16000.
- number of channels to one (mono).
- precision to 16-bit.

All data for training and evaluating should be in "TrainingData" with sub-directories as named <class ID>.

### 2. Structure

This step is done in the `data_init.py` script as well. 

By structuring, I mean splitting data for training and testing (80%, 20%)

The output of this step can be found at `audio` directory inside the `results` directory defined in the configuration file.

### 3. Feature Extraction

The file responsible for the feature extraction is `feature_utils.py` in which I extract features from the preprocessed audio files and save them into a new folder called `feature` represented by `FEATURE_DIR` config variable.

This process uses the following variables:

- `cepstral_coefficients`: the number of cepstral coeffiecients to be used when applyig MFCC.
- `filter_bank`: The type of filter-banknig used. It can be either `log`: logarithmic or `lin`:linear.
- `filter_bank_size`: Size of the filter bank.
- `lower_frequency`: the lower frequency (in Hz) of the filter bank.
- `higher_frequency`: the higher frequency (in Hz) of the filter bank.
- `vad`: The Voice Activity Detection algorithm used. It can be either "energy", "snr", "percentil" or "lbl".
- `snr_ratio`: The Signal-to-Noise ratio of the SNR algortihm (in db). It's useful only if "snr" is chosen as a vad algorithm.
- `window_size`: the size of the window for cep features.
- `window_shift`: The step that the window is moving (in sec).

The output of this step can be found at `feature` directory inside the `result` directory defined in the configuration file.

### 4. Choosing Model

In Sidekit, there are different models that we can train. I haven't been able to implement all the models, but the following are the ready ones:

- UBM: This is a Universal Background Model. You can modify the `num_gaussians` option in the config file.
- i-vector: This is an Identity Vector model. You can modify these configurations:
	- `batch_size`: the size of batch used for training i-vectors.
	- `tv_rank`: the rank (size) of the Total Variability matrix.
	- `tv_iterations`: number of iterations to train Total Variability matrix.


### 5. Train

`python3 data_init.py` 

`python3 feature_utils.py` 

`python3 ubm.py` 

`python3 iVector.py`

`python3 verify.py`

