# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Model type: Random Forest Classifier

Framework: Scikit-Learn

Trained with: Python 3.10 using train_model.py

Encoders: OneHotEncoder for categorical variables and LabelBinarizer for the target

Primary goal: Predict whether an individual earns >50K or <=50K annually based on U.S. Census demographic data

Owner / Developer: Devin Black

Date created: November 2025
## Intended Use
This model is designed for educational and experimental use within Udacity’s “Deploying a Scalable ML Pipeline with FastAPI” project.
It demonstrates:

how to build, train, and evaluate a supervised learning model,

how to compute fairness-oriented metrics on data slices, and

how to deploy a model using a RESTful API.

Not intended for production or real-world income predictions.
## Training Data
Source: UCI Adult Census Dataset
 (provided as data/census.csv)

Samples: ≈32,000 rows

Features: 14 attributes (e.g., age, education, occupation, hours-per-week, sex, race)

Target: salary (<=50K or >50K)

Preprocessing: Missing values handled; categorical variables one-hot encoded; labels binarized.
## Evaluation Data
20% of the dataset held out for testing.

Same preprocessing pipeline applied using saved encoder and label binarizer.
## Metrics
_Please include the metrics used and your model's performance on those metrics._
Metric	Value
Precision	0.7088
Recall	0.2717
F1-score	0.3928

To ensure fairness across subpopulations, metrics were computed per categorical feature slice (workclass, education, sex, race, etc.).
Performance was generally consistent, but lower recall was observed for smaller groups (e.g., “Never-worked”, “Other-race”), indicating potential under-representation.
## Ethical Considerations
Sensitive features such as sex, race, and native-country may encode social or economic biases.

The model’s predictions should not be used for decisions affecting real people.

Continuous monitoring and bias audits are recommended if used beyond demonstration.
## Caveats and Recommendations
For any operational use, retrain the model on current, representative data.

Consider fairness techniques (reweighting, balanced sampling).

Validate predictions periodically to monitor drift and bias.