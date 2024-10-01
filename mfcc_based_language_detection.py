import librosa
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier


def extract_mfcc(file_path, n_mfcc=13, sr=16000):
    """
    Extracting MFCC features from an audio file.

    Parameters:
    file_path (str): path to the audio file
    n_mfcc=13 (int): number of MFCC features to be computed
    sr=16000 (int): sampling rate
    Returns:
    mfcc_mean (np.array): an array of 13 MFCC features
    """

    # Loading the file
    y, sr = librosa.load(file_path, sr=sr)

    # Setting the length to 5 seconds, padding with zeros if > 0, and trimming if < 0.
    y = librosa.util.fix_length(y, size=5*sr)
    
    """
    The code is adjusted to produce the results of the original setup of the extraction of MFCCs. In order to repeat the experiments conducted, you need uncomment relevant lines and comment the irrelevant ones.
    """
    # Extracting the mel spectrogram, params for windowing, overlap, and mel bands are default (and best performing).
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
    
    # Converting the mel spectrogram to the log form, better representation of human perception.
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    
    # Computing the MFCC features.
    mfcc = librosa.feature.mfcc(S=log_mel_spectrogram, sr=sr, n_mfcc=n_mfcc)
    
    # Taking the mean.
    mfcc_mean = np.mean(mfcc.T, axis=0)
    
    # The below line is used to extract MFCCs from a non-framed audio sample. If you uncomment it, make sure you comment out all of the above lines.
    # mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc= n_mfcc)

    # # Computing the delta features (first-order derivative of MFCCs)
    # delta_mfcc = librosa.feature.delta(mfcc)

    # # Computing the delta-delta features (second-order derivative of MFCCs)
    # delta2_mfcc = librosa.feature.delta(mfcc, order=2)

    # # Merging MFCCs, Delta, and Delta-Delta features into one array
    # combined_features = np.vstack([mfcc, delta_mfcc, delta2_mfcc])

    # combined_features_mean = np.mean(combined_features.T, axis = 0)

    mfcc_mean = np.mean(mfcc.T,axis=0)

    # Original return
    return mfcc_mean
    #Return for delta and delta-delta features from Experiment 4.
    # return combined_features_mean
    # Excluding the first element, used in Experiments 1 and 3
    # return mfcc_mean[1:]


def preprocessing(directory):
    """
    Reading from a directory, obtaining MFCC features for each audio file, adding them to a dataset for a given
    language. Then concatenating each list with a label, thus creating a tuple.

    Parameters:
    directory (str): directory for a folder with audio files.
    Returns:
    dataset (list): list of tuples, each having MFCC at pos 0, and labels (classes) at pos 1.

    """
    features = []
    dataset = []
    label = directory[:2]

    for file in os.listdir(directory):
        # Avoiding an issue with files in an inappropriate format
        if file == '.DS_Store':
            continue
        filepath = directory+'/'+file
        features.append(extract_mfcc(filepath))

    for array in features:
        data_instance = (array,label)
        dataset.append(data_instance)
    return dataset


# Paths to train data
path_pl_train = "pl/train"
path_ru_train = "ru/train"
path_uk_train = "uk/train"
path_sk_train = "sk/train"

# Paths to test data
path_pl_test = "pl/test"
path_ru_test = "ru/test"
path_uk_test = "uk/test"
path_sk_test = "sk/test"

# Labels in order
labels= ['pl','ru','uk','sk']

# Preparing train instances
pl_train = preprocessing(path_pl_train)
ru_train = preprocessing(path_ru_train)
uk_train = preprocessing(path_uk_train)
sk_train = preprocessing(path_sk_train)

# Merging train instances
all_train_data = pl_train + ru_train + uk_train + sk_train

# Preparing test instances
pl_test = preprocessing(path_pl_test)
ru_test = preprocessing(path_ru_test)
uk_test = preprocessing(path_uk_test)
sk_test = preprocessing(path_sk_test)

# Merging test instances
all_test_data = pl_test + ru_test + uk_test + sk_test

# Initialising two lists, one to store MFCC arrays for training, and the other to store the corresponding labels
X_train = []
y_train = []

# Iterating throught the train data and filling out X_train and y_train
for i in all_train_data:
    X_train.append(i[0])
    y_train.append(i[1])

# Initialising two lists, one to store MFCC arrays for testing, the other to store the corresponding labels to compare against at inference time
X_test = []
y_test = []

# Iterating throught the test data and filling out X_train and y_train
for i in all_test_data:
    X_test.append(i[0])
    y_test.append(i[1])

"""
All three models are below. When running one of them, make sure the other two are commented out. 
"""
# # SVM model
# clf = svm.SVC(kernel='linear')
# clf.fit(X_train, y_train)

# y_pred = clf.predict(X_test)


# KNN model 
# knn9 = KNeighborsClassifier(n_neighbors=9)
# knn9.fit(X_train, y_train)
# y_pred = knn9.predict(X_test)


# RF model
clf = RandomForestClassifier(n_estimators=100, random_state=57)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

# Printing accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Showing the results in the form of a confusion matrix 
cm = confusion_matrix(y_test, y_pred)
print(cm)
cmp = ConfusionMatrixDisplay(cm, display_labels=labels)
fig, ax = plt.subplots(figsize=(8,8))
cmp.plot(ax=ax)
plt.show()