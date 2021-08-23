
# Mod 17 –Machine Learning & Decision Trees – Credit Risk Analysis
### Overview of Analysis 
The purpose of this analysis was to determine credit risk through the use of supervised machine learning algorithms from a dataset from LendingClub. The dataset is unbalanced as good loans outnumber risky loans by a significant portion. To correct these imbalances and provide us with appropriate predictive capabilities we used Python’s imbalanced-learn & scikit-learn models to determine which if any could successfully predict low and high-risk credit applications.  Resampling algorithms used were a combination of oversampling, undersampling, and mix of the two—specifically: RandomOverSampler, SMOTE, ClusterCentroids, and SMOTEENN. We also used the BalancedRandomForestClassifier and EasyEnsembleClassifier machine learning models to predict credit risk and reduce bias. The analysis below will detail the balanced accuracy scores, precisions, and sensitivity/recall scores for each model.

### Results
## RandomOverSampler

![RandomOverSampler_Accuracy](https://github.com/RichelynScott/Credit_Risk_Analysis/blob/main/Module-17-Challenge-Resources/images/RandomOverSampler_Accuracy.png)
![RandomOverSampler_ICR](https://github.com/RichelynScott/Credit_Risk_Analysis/blob/main/Module-17-Challenge-Resources/images/RandomOverSampler_imbalanced_Classification_Report.png)
![RandomOverSampler_CM](https://github.com/RichelynScott/Credit_Risk_Analysis/blob/main/Module-17-Challenge-Resources/images/RandomOverSampler_CM.png)

•	The ROS Model has a moderate accuracy of about 65.5%

•	The ROS Model has an extremely high precision of about 100% for detecting low-risk applicants but an abysmal 1%  for predicting high-risk applicants

•	The ROS Model had a modest Sensitivity/Recall score of 72% for high-risk applicants and a mediocre Sensitivity/Recall score of 59% for low-risk applicants

## SMOTE

![SMOTE_Accuracy](https://github.com/RichelynScott/Credit_Risk_Analysis/blob/main/Module-17-Challenge-Resources/images/SMOTE_Accuracy.png)
![SMOTE_ICR](https://github.com/RichelynScott/Credit_Risk_Analysis/blob/main/Module-17-Challenge-Resources/images/SMOTEENN_ICR.png)
![SMOTE_CM](https://github.com/RichelynScott/Credit_Risk_Analysis/blob/main/Module-17-Challenge-Resources/images/SMOTEENN_CM.png)

•	The SMOTE Model had a moderate accuracy (little better than the ROS Model) of about 66.2%

•	The SMOTE Model has an extremely high precision of about 100% for detecting low-risk applicants but an abysmal 1%  for predicting high-risk applicants (Just like the ROS Model)

•	The SMOTE Model had a modest Sensitivity/Recall score of 69% for high-risk applicants and a modest Sensitivity/Recall score of 63% for low-risk applicants (Slightly higher than the ROS Model)

## Cluster Centroids Undersampling Model

![CC_Accuracy](https://github.com/RichelynScott/Credit_Risk_Analysis/blob/main/Module-17-Challenge-Resources/images/ClusterCentroids_Accuracy.png)
![CC_ICR](https://github.com/RichelynScott/Credit_Risk_Analysis/blob/main/Module-17-Challenge-Resources/images/ClusterCentroids_ICR.png)
![CC_CM](https://github.com/RichelynScott/Credit_Risk_Analysis/blob/main/Module-17-Challenge-Resources/images/ClusterCentroids_CM.png)

•	The Cluster Centroids Model had a lower accuracy than the ROS & SMOTE models of about 54.5%

•	The Cluster Centroids Model has an extremely high precision of about 100% for detecting low-risk applicants but an abysmal 1% for predicting high-risk applicants (Just like the ROS & SMOTE Models)

•	The Cluster Centroids Model had a modest Sensitivity/Recall score of 69% for high-risk applicants and a poor Sensitivity/Recall score of 40% for low-risk applicants (The worst of all current models thus far)


