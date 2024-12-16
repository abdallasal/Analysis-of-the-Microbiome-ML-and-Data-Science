Python project using machine learning methods to build a pipeline that identifies individuals with an unfamiliar disease based on their gut microbiome composition.
The pipeline input is a gut microbiome composition csv file and meatadata csv file containing information about patients.
The pipeline processes the data by removing low variance features and using feature selection methods (MRMR feature selection).
Then it splits the data into train and test sets to train the Machine Learning model (Random Forest Classsifier), using 2 methods, 70-30 split, and Leave-One-Disease-Out.
70-30 method helps the model predict patients with previously trained on diseases, and the Leave-One_disease-Out helps the model predict unseen diseases.
The pipeline outputs the probability that each individual is not healthy.
