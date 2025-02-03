# -Bank-Customer-Churn-Prediction-Using-Deep-Learning-TensorFlow-Keras-ANN-

#  Objective

This project focuses on predicting customer churn for a bank using deep learning techniques. Customer churn prediction helps businesses identify customers at risk of leaving, enabling proactive retention strategies. The implementation leverages **TensorFlow**, **Keras**, and an **Artificial Neural Network (ANN)** to build and evaluate a binary classification model.

# Dataset
Source: Kaggle (Bank Customer Churn Modeling dataset)

Features: Include customer demographics, account balance, tenure, and transaction behavior.

Target variable: Exited (1 = Churn, 0 = Retained)

# Approach
# Data preprocessing.
Removed unnecessary columns (RowNumber, CustomerId, Surname).

Checked for missing values and ensured data consistency.

Encoded categorical variables (Gender using label encoding, Geography using one-hot encoding)

Feature Scaling: Standardizes numerical features with StandardScaler for optimal model performance.

Separated features (X) and target (y)
# Model Architecture
A Sequential ANN is built with the following layers:

Input Layer: 12 features (after preprocessing).

Hidden Layers: Two dense layers with ReLU activation and one output layer.

Output Layer: Sigmoid activation for binary classification.

# Training
Optimizer: Adam.

Loss Function: Binary cross-entropy.

Early Stopping: Monitors validation loss to prevent overfitting.

Training: 100 epochs with a validation split of 20%.

# Evaluation
Accuracy: Model performance on test data.

Confusion Matrix: Visualizes true vs. predicted labels.

Classification Report: Provides precision, recall, and F1-score

# Results
The model achieves ~85-86% accuracy on the test set.

Key metrics (precision, recall) highlight performance on the minority class (churned customers).

Example prediction on a sample customer demonstrates practical usage.

After checking the results the F1-score is low so, i checked the classed in the target variable so the class is imbalances.

By using the SMOOTH technique the feature is balanced.

Now the whole process is repeted again with the same ANN structed.

now F1-score is also balanced.

# Benefits of SMOTE:
Improved recall for the minority class (churned customers).

Better generalization due to diverse synthetic samples.

More reliable precision-recall trade-off.

# Conclusion
Customer churn prediction is a critical challenge for businesses aiming to retain clients and sustain growth. In this project, leveraging deep learning with TensorFlow/Keras and an ANN architecture provided a robust framework for identifying at-risk customers. However, the inherent class imbalance in churn datasets—where retained customers vastly outnumber those who churn—poses a significant hurdle, often leading models to prioritize majority-class accuracy while neglecting the critical minority class.

The integration of SMOTE (Synthetic Minority Oversampling Technique) addressed this imbalance by generating synthetic samples of churned customers, enabling the model to learn nuanced patterns associated with attrition. This approach not only improved recall for the minority class (ensuring more churners are correctly identified) but also maintained a balanced precision-recall trade-off, making predictions actionable for business strategies.

By deploying such a model, banks can proactively engage customers with personalized retention efforts—such as targeted offers, improved customer service, or loyalty programs—directly reducing attrition costs and enhancing customer lifetime value. While SMOTE significantly enhanced model performance, future work could explore hybrid techniques (e.g., SMOTE with undersampling) or advanced architectures like graph neural networks to further refine predictions.

Ultimately, this project underscores the power of combining deep learning with strategic data preprocessing to solve real-world business challenges. By prioritizing both technical rigor and practical applicability, organizations can transform raw data into actionable insights, fostering long-term customer relationships and driving sustainable success.


