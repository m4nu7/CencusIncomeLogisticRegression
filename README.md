## About this project

This is an end to end Logistic Regression ML project.

This is a project to predict whether a person makes over or under $50k a year.

## Dataset : Cencus Income [https://archive.ics.uci.edu/dataset/20/census+income]
This data was extracted from the census bureau database found at
http://www.census.gov/ftp/pub/DES/www/welcome.html

|Donor: Ronny Kohavi and Barry Becker
|        Data Mining and Visualization
|        Silicon Graphics.
|        e-mail: ronnyk@sgi.com

 - 48842 instances, mix of continuous and discrete    (train=32561, test=16281)


## Dataset features :


|  Variable Name |    Role  |       Type |                                       Description |Units | Missing Values|
|----------------|----------|------------|---------------------------------------------------|------| --------------|
|            age | Feature  |    Integer | age of an individual                              |  NaN |             no|
|      workclass | Feature  |Categorical | employment status of an individual                |  NaN |            yes|
|         fnlwgt | Feature  |    Integer | final weight. This is the numberof people the     |  NaN |             no|
|                |          |            | census believes the entry represents              |      |               |
|      education | Feature  |Categorical | Individuals highest level of education achieved   |  NaN |             no|
|  education-num | Feature  |    Integer | highest level of education achieved in numerical form|  NaN |             no|
| marital-status | Feature  |Categorical | marital status of an individual                   |  NaN |             no|
|     occupation | Feature  |Categorical | general type of occupation of an individual       |  NaN |            yes|
|   relationship | Feature  |Categorical | represents what this individual is relative to others |  NaN |             no|
|           race | Feature  |Categorical | Descriptions of an individual’s race              |  NaN |             no|
|            sex | Feature  |     Binary | the sex of the individual                         |  NaN |             no|
|   capital-gain | Feature  |    Integer | capital gains for an individual                   |  NaN |             no|
|   capital-loss | Feature  |    Integer | capital loss for an individual                    |  NaN |             no|
| hours-per-week | Feature  |    Integer | the hours an individual has reported to work per week |  NaN |             no|
| native-country | Feature  |Categorical | country of origin for an individual               |  NaN |            yes|
|         income |  Target  |     Binary | income of an individual (Binary: >50K, <=50K)     |  NaN |             no|


### Created a new environment

```
conda create -p venv python=3.8
conda activate venv/
```

### Install all necessary libraries
```
pip install -r requirements.txt
```

