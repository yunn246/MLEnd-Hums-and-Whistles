# MLEnd-Hums-and-Whistles

# Problem formulation

The MLEnd Hums and Whistles Dataset is created by the students of Queen Mary University, and we know the dataset is balanced. However, this might not always be the case for many other datasets. For example, a person's mailbox has ham emails and spam emails. The number of them is often imbalanced. 

The problem formulated is a supervised classification problem. In this problem, an imbalanced dataset is loaded for building the model. The total number of sample songs is 446, but only 30 are "Panther", the others are "Hakuna". The model is built with data to predict the labels, and we can check if it can correctly classify both songs.

# Machine Learning pipeline

The raw data for the machine learning pipeline is the .wav song files. 

**1. Input**

Before the files are used to train the model, they must be pre-processed. The intermediate stage is feature extraction. The extracted features are the input to the model.

**2. Model**

Some models are tried and compared. They are evaluated by the output results. 

**3. Output**

The predicted labels are the outputs, and the outputs are used to assess the model quality.

The machine learning pipeline includes feature extraction and the model. If there is a deployment stage, the whole pipeline will be deployed.

# Preprocessing
The raw data are .wav audio files. The Librosa package is used to read the files into the notebook. For some feature extraction processes, the songs need to have the same length to obtain the same number of features, but not all of them precisely follow the 15 seconds standard. Hence, some durations are set to a different length when loading the files. Some people keep a silent gap before humming in the recordings, whereas others start to hum or whistle straight away. We can also hear the voice of clicking a button or a mouse, which should be considered noise in this case. To avoid that, the files are read from the 2nd second, and the durations are less than the actual length of the songs. This would help to skip the clicking sound and not extract features of unnecessary parts.

# Transformation

The result of FFT contains too many features, so the dimension reduction method is used in the transformation stage to make sure the transformed data is the correct data type to be input for the model. Principle component analysis is used for the Fast Fourier Transform result to change the number of features from 174195 to 3.

# Feature exraction and selection

It is getting all the features ready and fitting them in the dataframe as input for the model. This step is important, and the index of each component need to be aligned. Otherwise, it cannot successfully be trained and predict labels; or it could predict wrong labels. 

# Modelling

Some classifiers are tried with a combination of different sets of attributes. Some features are extracted and used to train first and gradually narrow down the selection by trying different combinations. The candidate classifiers for the experiment include Gaussian Naive Bayes, SVM, logistic regression and random forests.

**Gaussian Naive Bayes** classifier follows Gaussian normal distribution. It is one of the Bayes classifiers in supervised learning, which considers the posterior probability, such as the proportion of each class. 

**Support Vector Machine(SVM)** works when the number of dimensions is higher than the number of samples. It takes each sample as a vector and creates a hyperplane to separate the classes. C-Support Vector Classification (SVC) is used here. It has a regularization parameter C=1 and a random state. A random seed needs to be set; otherwise, the result will vary in every run.

**Logistic regression** is a procedure to build a linear classifier. It calculates the distance from samples to a boundary, representing the certainty of a sample in the labelled class.

**Random forests** is an ensemble methodology for classification. In a decision tree, we want to avoid impurity, and the split starts from the root with one attribute at a time. Random forests combine multiple decision trees. 

# Methodology

Unlike a balanced binary classification problem, accuracy might not be the most suitable quality metric for the classifier. In this case, the confusion matrix is a more proper way to visualize the result and the macro average of f1-score to measure the performance of each classifier and feature.

The accuracy might be high for the model because the dataset is imbalanced. If a classifier predicts all the samples belong to the majority class, the accuracy can be high. However, this does not mean the classifier is precisely classifying the samples. Instead, it mispredicts all samples in the minority group. This is why the confusion matrix and classification report are more appropriate for assessing the quality of this problem.

The preprocessed feature will be trained on the training set, and validated on the validation set. The test set cannot be used until a final model is selected. The proportion of the minority in each subset is around 7%. 




