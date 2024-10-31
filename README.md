# US Visa Approval Prediction

`MLOps Production Ready Machine Learning Classification Project On Visa Approval Prediction`

# Project Overview

### 1. Understanding the Problem Statement

Given certan set of feature such as (continent, education, job_experience, training, employment, current age etc). On the basis of these data I have to predict `whether the application for the visa will be approved or not`

Features:

- Continent: Asia, Africa, North America, Europe, South America,

- Oceania Eduction: High School, Master's Degree, Bachelor's, Doctorate

- Job Experience: Yes,

- Required training: Yes, No

- Number of employees: 15000 to 40000

- Region of employment: West, Northeast, South, Midwest, Island

- Prevailing wage: 700to 70000

- Contract Tenure: Hour, Year, Week, Month

- Full time Yes, No

- Age of company: 15 to 180

### 2. Understanding the Solution

#### Solution scope:

1. This can be used on real life by US Visa Applicants so that they can improve their `Resume` and `Criteria` for the approval process.

2. Also this model can be used by US Visa officers to filter the candidate for visa approval automatically with involvement of much manual work or manpower, this overcome the manual task.

#### Solution Approch:

1. Machine Learning: ML Classification Algorithms
2. Deep Learning: Custom ANN with sigmoid activation function

#### Solution Procosed:

I will be using Machine Learning approcah to solve this problem.

1. Load the dataset from Database
2. Perform EDA and Feature Engineering to select the desirable features.
3. Fit the ML Classification Algorithms and find out which one performs better.
4. Select top performing few and tune with hyperparameters like Gridsearch CV.
5. Select the best model based on desired metrices.

### 3. Code Understanding and walkthrough

Use OOPs concept to make code clean and DRY

### 4. Understanding the Deployment

1. Docker
2. Cloud Services like AWS
3. Adding self Hosted Runner
4. MLFlow
5. Data Version Control (DVC)
6.
