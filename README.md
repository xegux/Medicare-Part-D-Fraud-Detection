# Anomaly Medicare Part D Fraud Detection Using Supervised and Unsupervised Learning

Mentor:  Dr. Janos Hajagos, Stony Brook University

Medicare fraud costs the US hundreds of billions of dollars each year and current efforts are insufficient in efficiently detecting fraud prescribers. 
The primary problem in the detection of fraud in Medicare Part D is that fraud providers only make up 0.000116% of all the providers. Thus far, no 
machine learning models in detecting Part D fraud have produced an area-under-the-curve (AUC) score of over 0.8. Utilizing training data augmentation, 
upsampling, and downsampling, eight classification algorithms were tested on the Part D data from 2013-2017, along with an LSTM RNN. Two exploratory 
data analytical methods were also conducted: distance matrices calculating the euclidean distance between two providers based on the similarity of their 
prescription patterns and market basket analysis detecting potential fraud based on pairs of drugs. The AUC scores of the machine learning models were 
significantly improved with the training data augmentation while the ROS, RUS, and ROS+RUS that previous studies used, had a much less positive effect. 
Surprisingly, the K-Nearest Neighbors algorithm performed the best with an AUC of 0.86, the highest out of any prior models in Medicare fraud detection. 
Additionally, an 80%-20% training/testing data split performed the best and that the inclusion of 188 specialty features and schedule 2 drug features 
improved each model’s AUC by an average of 0.04. Finally, the RNN for the top 4 fraud states yielded an AUC of 0.82, a big triumph as it is the first 
sequential Medicare fraud machine learning model. 

