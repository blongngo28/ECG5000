# ECG5000
Bachelor Project : 
1. What this project does ?

    In this project , I have successfully reimplemented the Time-Series Memory Augmented Autoencoder(TSMAE) model from a famous paper published in 2023 referred in [1]. The special thing and the significant reason I want to reimplemented it since the complexity of the model. As we know , anomaly detection in IoT Time series data is a challenge mission due to high-dimensional and heterogenous nature data and many traditional machine learning models such as Isolation Forest, One-Class Support Vector Machine or Vanilla Autoencoder cannot handle it efficiently . The discovery of TSMAE not only achieves high performance in evaluation metric but also contributes a roburt solution for anomaly detection in IoT application.

2. Why this project is useful ?

- Beginner students can learn about some basics neural networks such as deep neural network or recurrent neural network like Long-short term memory and Autoencoder.
- Advanced students can understand why this state-of-the-art model is more effective than many traditional models.

3. How user can started with the project ?

 - Step 1 : Understand some basic mathematical concepts and knowledges in AI and then try to code some basic math in Python
 - Step 2 : Learn some machine learning models and available library which are useful for anomaly detection mission
 - Step 3 : Read the paper and investigate what this paper contributes to our AI society
 - Step 4 : Try to code it by yourself before read the code files in here.

4. Datasets : The datasets taken from paper taken from this link here : https://www.kaggle.com/datasets/salsabilahmid/ecg50000/code

5. Datasets description :

   - ECG5000 dataset : 5000 sequences of electrocardiogram signal with 140 features , 58.4% normal heartbeat and 41.6% abnormal heaatbeat
   -  Wafer dataset : originates from semiconductor manufacturing processes and contain sensor measurements recorded during
   the production of silicon wafer
   - Arrhythmya dataset : is designed to analyze ECG readings for identifying and classifying cardiac arrhythmias. The dataset
     includes two main groups : Class 01, representing normal ECG data and class 02 throug 16 , which encompass various types of arrhythmia and unknowncases.

6. Related papers :
   [1] H. Gao, B. Qiu, R. J. D. Barroso, W. Hussain, Y. Xu and X. Wang, "TSMAE: A Novel Anomaly Detection Approach for Internet of Things Time Series Data Using Memory-Augmented Autoencoder," in IEEE Transactions on Network Science and Engineering, vol. 10, no. 5, pp. 2978-2990, 1 Sept.-Oct. 2023, doi: 10.1109/TNSE.2022.3163144.
keywords: {Anomaly detection;Data models;Task analysis;Feature extraction;Internet of Things;Decoding;Complexity theory;Anomaly detection;time-series classification;memory augmentation;autoencoder},



