# Mod 17 –Machine Learning & Decision Trees – Credit Risk Analysis
### Overview of Analysis 
The purpose of this analysis was to determine credit risk through the use of supervised machine learning algorithms from a dataset from LendingClub. The dataset is unbalanced as good loans outnumber risky loans by a significant portion. To correct these imbalances and provide us with appropriate predictive capabilities we used Python’s imbalanced-learn & scikit-learn models to determine which if any could successfully predict low and high-risk credit applications.  Resampling algorithms used were a combination of oversampling, undersampling, and mix of the two—specifically: RandomOverSampler, SMOTE, ClusterCentroids, and SMOTEENN. We also used the BalancedRandomForestClassifier and EasyEnsembleClassifier machine learning models to predict credit risk and reduce bias. The analysis below will detail the balanced accuracy scores, precisions, and sensitivity/recall scores for each model.

### Results
## RandomOverSampler

![RandomOverSampler]( 
