import math
import pandas as pd
from sklearn.feature_selection import f_regression
from sklearn.ensemble import RandomForestClassifier

# K is the number of features to select in feature selection section, 3 one hot encoding features will be added,
# overall features to be selected: 28.
K = 25

# initializing the random forest classifier with hyperparameters tuned according to grid search cross validation
mod = RandomForestClassifier(n_estimators=500, max_depth=10,
                             min_samples_split=20, min_samples_leaf=1,
                             min_weight_fraction_leaf=0.0, max_features='sqrt', max_leaf_nodes=None,
                             bootstrap=True, oob_score=True, n_jobs=None, random_state=0, verbose=0)

# training files paths
etc_metadata_file_path = r"\etc\metadata.txt"
etc_specie_file_path = r"\etc\mtx_specie.txt"

#input files' paths
input_metadata_path = r"\input\metadata-20230524T113718Z-001\metadata\metadata.txt"
input_specie_path = r"\input\omic_data-20230524T113720Z-001\omic_data\mtx_specie.txt"

#output file path
output_path = r"\output\output.csv"


# importing the data sets and saving them as data frames, metadata df and specie (specie-level) df
def df_init(metadata_file_path, specie_file_path):
    # Metadata df
    metadata_df = pd.read_csv(metadata_file_path, sep='\t')
    metadata_df.replace("2a", "2", inplace=True)
    metadata_df["PatientGroup"] = metadata_df["PatientGroup"].astype(int)

    # Omics df - species
    specie_df = pd.read_csv(specie_file_path, delim_whitespace=True)
    return metadata_df, specie_df


# removing unwanted columns from the specie df, columns containing over 95% zeros, thus very low variance
def specie_pre_processing(specie_df):
    non_zero_columns_cnt = pd.DataFrame(specie_df.astype(bool).sum(axis=0))
    math.ceil(0.05 * len(specie_df))

    # Removing columns with 95% or more of its values are zeros
    flt = non_zero_columns_cnt[non_zero_columns_cnt[0] >= (math.ceil(0.05 * len(specie_df)))]
    specie_df2 = specie_df[flt.index]
    return specie_df2


# mrmr feature selection method, code source: Mazzanti, S. "MRMR" explained exactly how you wished
# someone explained to you. Towards Data Science (2021).
# https://towardsdatascience.com/mrmr-explained-exactly-how-you-wished-someone-explained-to-you-9cf4ed27458b
def mrmr_fs(X, y, K):
    # compute F-statistics and initialize correlation matrix
    F = pd.Series(f_regression(X, y.values.ravel())[0], index=X.columns)
    corr = pd.DataFrame(.0001, index=X.columns, columns=X.columns)

    # initialize list of selected features and list of excluded features
    selected = []
    not_selected = X.columns.to_list()

    # repeat K times
    for i in range(K):
        # compute (absolute) correlations between the last selected feature and all the (currently) excluded features
        if i > 0:
            last_selected = selected[-1]
            corr.loc[not_selected, last_selected] = X[not_selected].corrwith(X[last_selected]).abs().clip(.0001)

        # compute FCQ score for all the (currently) excluded features (this is Formula 2)
        score = F.loc[not_selected] / corr.loc[not_selected, selected].mean(axis=1).fillna(.0001)

        # find best feature, add it to selected and remove it from not_selected
        best = score.index[score.argmax()]
        selected.append(best)
        not_selected.remove(best)
    return selected


# helper function, marking patients as 1 and healthy as 0
def sick_control(val):
    # Healthy is 0
    if val == 8:
        return 0
    # Sick is 1
    else:
        return 1


# feature selection function, merging both metadata df and specie df (specie df is the output of the function
# specie_pre_processing), and selecting K+3 features.
def feature_selection(metadata_df, specie_df, K):
    sample_pg = metadata_df[["SampleID", "PatientGroup", "CENTER"]]
    fs_df = pd.merge(specie_df, sample_pg, on="SampleID", how="inner")
    fs_df["Patient"] = fs_df["PatientGroup"].apply(sick_control)
    fs_df2 = fs_df.drop(columns=["SampleID", "PatientGroup", "CENTER"])

    X = fs_df2.iloc[:, :-1]
    y = fs_df2.iloc[:, -1]
    features = mrmr_fs(X, y, K)

    all_feats = features.copy()
    all_feats.append("SampleID")
    all_feats.append("Patient")
    all_feats.append("PatientGroup")
    all_feats.append("CENTER")
    df = fs_df[all_feats]

    ohe_center = pd.get_dummies(df["CENTER"])
    ohe_center_numeric = ohe_center.astype(int)
    df = pd.concat([df, ohe_center_numeric], axis=1)
    df.drop(columns=["SampleID", "CENTER"], inplace=True)
    return df


# splitting the df after feature selection into 70-30 train-test, while preserving the percent of controls
# and patients in both train and test as the original percent in the primary df
def train_test_split(df):
    df = df.drop(columns=["PatientGroup"])
    df_patients = df[df["Patient"] == 1]
    df_controls = df[df["Patient"] == 0]
    train_num_patients = int(len(df_patients) * 0.70)
    train_num_controls = int(len(df_controls) * 0.70)

    train_patients_sampled = df_patients.sample(n=train_num_patients, random_state=0)
    test_patients_sampled = df_patients.drop(train_patients_sampled.index)

    train_controls_sampled = df_controls.sample(n=train_num_controls, random_state=0)
    test_controls_sampled = df_controls.drop(train_controls_sampled.index)

    test_patients_sampled_label = test_patients_sampled["Patient"]
    test_patients_sampled_data = test_patients_sampled.drop(columns=["Patient"])

    test_controls_sampled_label = test_controls_sampled["Patient"]
    test_controls_sampled_data = test_controls_sampled.drop(columns=["Patient"])

    test_data = pd.concat([test_patients_sampled_data, test_controls_sampled_data], axis=0)
    test_label = pd.concat([test_patients_sampled_label, test_controls_sampled_label], axis=0)

    train_and_label = pd.concat([train_patients_sampled, train_controls_sampled], axis=0)
    train_data = train_and_label.drop(columns=["Patient"])
    train_label = train_and_label["Patient"]
    return train_data, train_label, test_data, test_label


# training the model after splitting the data into 70-30 train test
def model_train1_70_30(train_data, train_label):
    mod.fit(train_data, train_label)
    return mod


# training the model in Leave-One_disease-Out method, first by processing the df given from the feature selection
# function, by splitting it into train-test the same split as before, and then using only the train set, which will be
# processed and split into subsets, each one will be excluded from one patient group with some controls to be the test
# and other will be the train set, and we will iterate over all patient groups.
def model_train2_lodo(df, test_data):
    lodo_df = df.copy()
    lodo_test = lodo_df.iloc[test_data.index]
    lodo_df.drop(test_data.index, inplace=True)

    pgs = list(lodo_df["PatientGroup"].unique())
    pgs.sort()
    pgs = pgs[:-1]

    lodo_pg_data = []
    lodo_pg_test = []

    lodo_test_controls = lodo_test[lodo_test["PatientGroup"] == 8]
    lodo_test_diseases = lodo_test[lodo_test["PatientGroup"] != 8]

    for pg in pgs:
        train_wo_pg = lodo_df[lodo_df["PatientGroup"] != pg]

        pg_test = lodo_test_diseases[lodo_test_diseases["PatientGroup"] == pg]
        relevant_test_controls_num = len(pg_test) / len(lodo_test_diseases)
        test_control = lodo_test_controls.sample(frac=relevant_test_controls_num)

        pg_test_w_ctrls = pd.concat([pg_test, test_control], axis=0)

        lodo_pg_data.append(train_wo_pg.drop(columns=["PatientGroup"]))
        lodo_pg_test.append(pg_test_w_ctrls.drop(columns=["PatientGroup"]))

    # model training
    for i in range(len(pgs)):
        train_label = lodo_pg_data[i]["Patient"]
        train_data = lodo_pg_data[i].drop(columns=["Patient"])
        mod.fit(train_data, train_label)
    return mod


# running the model training pipeline
def model_ready():
    # import the training data frames
    metadata_df, specie_df = df_init(etc_metadata_file_path, etc_specie_file_path)

    # specie data frame after pre-processing
    specie_df2 = specie_pre_processing(specie_df)

    # combining metadata and specie dfs into one df, and selecting K features while adding the OHE 3 features,
    # overall K+3 features. df is the final df after feature selection.
    df = feature_selection(metadata_df, specie_df2, K)

    # splitting df into 70-30 train-test
    train_data, train_label, test_data, test_label = train_test_split(df)

    # train the model take #1, 70-30 train-test split
    model_train1_70_30(train_data, train_label)

    # train the model take #2, Leave-One-Disease-Out approach
    model_train2_lodo(df, test_data)
    return df


# running the input - output processing, given the test set in the input, and delivering the probabilities in
# the output. This function also deals with features found in the training section, but not in the test set.
def input_output(df):
    # input dfs
    input_metadata_df, input_specie_df = df_init(input_metadata_path, input_specie_path)
    input_df = pd.merge(input_specie_df, input_metadata_df, on="SampleID", how="inner")
    input_df.drop(columns=["Patient"], inplace=True)
    ids = input_df["SampleID"]

    # set the input df features
    features = df.columns.tolist()
    input_features = input_df.columns.tolist()
    if "Patient" in features:
        features.remove("Patient")
    if "PatientGroup" in features:
        features.remove("PatientGroup")

    if "CENTER" in input_features:
        ohe_center = pd.get_dummies(input_df["CENTER"])
        ohe_center_numeric = ohe_center.astype(int)
        input_df = pd.concat([input_df, ohe_center_numeric], axis=1)
        input_df.drop(columns=["SampleID", "CENTER"], inplace=True)

    # add missing features from df to input_df, set their values to zero
    for feature in features:
        if feature not in input_features:
            # add missing feature with all values set to 0
            input_df[feature] = 0

    # order columns of input_df as the df
    input_df = input_df[features]

    # prediction
    prob_pred = mod.predict_proba(input_df)
    sick_prob = prob_pred[:, 1]

    # output creation
    output_df = pd.DataFrame()
    output_df["ID"] = ids
    output_df["Probability"] = sick_prob

    # output export
    output_df.to_csv(output_path, index=False)
    return


# running the model and dealing with input-output
df = model_ready()
input_output(df)