# mccc_based_language_detection
A small project on the implementation of classic ML approaches, namely, K-nearest neighbours, Random Forest, and Support Vector Machines, in MFCC-based language detection.

The project experiments with various ways in which MFCC feature extraction technique can be implemented when identifying a language from among languages that are phonologically and morphologically related, using the case of Polish, Russian, Slovak, and Ukrainian languages. The experiments conducted are as follows:

1. Taking the original MFCC extraction setup with the global mean value of MFCCs across all frames and removing C0 (C0 captures the overall amplitude of the audio recording and is associated with the loudness of the recording, which may be disregarded when working with language identification),thus leaving an array of 12 MFCCs as the vector representation of the audio files.
2. The duration of each audio file was set to 5 seconds at the time of data pre-processing, which is sufficient for extracting MFCCs of the entire recording, rather than extracting them from each frame and taking the mean. This approach informs the second experiment, with the preservation of C0 in the array.
3. Combining the previous two experiments by virtue of extracting MFCCs from the entire audio file and excluding C0.
4. Incorporating delta and delta-delta features in the initial array of MFCCs.
