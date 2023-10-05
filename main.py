import numpy as np
import pandas as pd
np.set_printoptions(suppress=True)
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)
pd.set_option('mode.chained_assignment', None)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost
import pickle
import streamlit as st

# Importing datasets
def read_train_data():
    # read the train.csv file using pandas library and return it
    train = pd.read_csv('train.csv')
    return train

def read_test_data():
    # read the test.csv file using pandas library and return it
    test = pd.read_csv('test.csv')
    return test

# NA values in train dataset
def check_null_values_train():
    train = read_train_data()
    # Check for null values using the isnull() method and sum them for each column
    train.isnull().sum()
    return train

# NA values in test dataset
def check_null_values_test():
    test = read_test_data()
    # Check for null values using the isnull() method and sum them for each column
    test.isnull().sum()
    return test

def filling_values_train():
    train = read_train_data()

    # Fill missing values in the 'Bed Grade' column with the most frequent value (mode)
    train['Bed Grade'].fillna(train['Bed Grade'].mode()[0], inplace=True)
    
    # Replace missing values in the 'City_Code_Patient' column with the most frequent value (mode)
    train['City_Code_Patient'].fillna(train['City_Code_Patient'].mode()[0], inplace=True)
    
    #Return the updated 'train' DataFrame with missing values filled
    return train

def filling_values_test():
    test = read_test_data()

    # Fill missing values in the 'Bed Grade' column with the most frequent value (mode)
    test['Bed Grade'].fillna(test['Bed Grade'].mode()[0], inplace=True)

    # Fill missing values in the 'City_Code_Patient' column with the most frequent value (mode)
    test['City_Code_Patient'].fillna(test['City_Code_Patient'].mode()[0], inplace=True)

    # SReturn the updated 'test' DataFrame with missing values filled
    return test

le = LabelEncoder()

def label_encoding():
    train = filling_values_train()

    # Perform label encoding on the 'Stay' column
    label_encoder = LabelEncoder()
    train['Stay'] = label_encoder.fit_transform(train['Stay'])

    # Return the updated 'train' DataFrame with the 'Stay' column label-encoded
    return train

#Imputing dummy Stay column in test datset to concatenate with train dataset
def dummy_column():
    test = filling_values_test()

    # Set a dummy value (-1) for the 'Stay' column in the 'test' DataFrame
    test['Stay'] = -1

    # Return the updated 'test' DataFrame with the 'Stay' column set to the dummy value
    return test

def concat():
    train = label_encoding()
    test = dummy_column()

    # Concatenate the 'train' and 'test' DataFrames along rows, ignoring their original indexes
    df = pd.concat([train, test], ignore_index=True)

    # Return the concatenated DataFrame
    return df

def label_encoding_df():
    df = concat()

    # Perform label encoding on selected categorical columns ('Hospital_type_code', 'Hospital_region_code', 'Department', 'Ward_Type', 'Ward_Facility_Code', 'Type of Admission', 'Severity of Illness', 'Age')
    # Define the list of selected categorical columns for label encoding
    categorical_columns = ['Hospital_type_code', 'Hospital_region_code', 'Department', 'Ward_Type', 'Ward_Facility_Code', 'Type of Admission', 'Severity of Illness', 'Age']

    # Perform label encoding on selected categorical columns
    label_encoder = LabelEncoder()
    for column in categorical_columns:
        df[column] = label_encoder.fit_transform(df[column])

    # Return the updated DataFrame with label-encoded categorical columns
    return df

def get_countid_encode(train, test, cols, name):
  temp = train.groupby(cols)['case_id'].count().reset_index().rename(columns = {'case_id': name})
  temp2 = test.groupby(cols)['case_id'].count().reset_index().rename(columns = {'case_id': name})
  train = pd.merge(train, temp, how='left', on= cols)
  test = pd.merge(test,temp2, how='left', on= cols)
  train[name] = train[name].astype('float')
  test[name] = test[name].astype('float')
  train[name].fillna(np.median(temp[name]), inplace = True)
  test[name].fillna(np.median(temp2[name]), inplace = True)
  return train, test

def get_countid_encode_test(test, cols, name):
  temp2 = test.groupby(cols)['case_id'].count().reset_index().rename(columns = {'case_id': name})
  test = pd.merge(test,temp2, how='left', on= cols)
  test[name] = test[name].astype('float')
  test[name].fillna(np.median(temp2[name]), inplace = True)
  return test

#Spearating Train and Test Datasets
def separating_train():
    df = label_encoding_df()

    # Separate the 'train' DataFrame by excluding rows with value (-1) in 'Stay' column
    train = df[df['Stay'] != -1]

    # Return the 'train' DataFrame containing only real training data
    return train

def separating_test():
    df = label_encoding_df()

    # Separate the 'test' DataFrame by selecting rows with value (-1) in 'Stay' column
    test = df[df['Stay'] == -1]

    # Return the 'test' DataFrame containing only the test data with dummy 'Stay' values
    return test

# Droping duplicate columns
def drop_duplicates_test(test):
    # Drop specified columns from the 'test' DataFrame ('Stay', 'patientid', 'Hospital_region_code', 'Ward_Facility_Code')
    columns_to_drop = ['Stay', 'patientid', 'Hospital_region_code', 'Ward_Facility_Code']
    test1 = test.drop(columns=columns_to_drop)

    # Return the 'test1' DataFrame with duplicate rows removed and specific columns dropped
    return test1

def drop_duplicates_train(train):
    # Drop specified columns from the 'train' DataFrame ('case_id', 'patientid', 'Hospital_region_code', 'Ward_Facility_Code')
    columns_to_drop = ['case_id', 'patientid', 'Hospital_region_code', 'Ward_Facility_Code']
    train1 = train.drop(columns=columns_to_drop)

    # Return the 'train1' DataFrame with duplicate rows removed and specific columns dropped
    return train1

# Splitting train data for XGBoost
def splitting(train):
    # do not edit the predefined function name
    train1 = drop_duplicates_train(train)

    # Separate features (X) and target variable (y) from the 'train1' DataFrame
    # For x select the data without stay column, for  y select only stay column
    X = train1.drop(columns=['Stay'])  # Features (exclude 'Stay' column)
    y = train1['Stay']  # Target variable (only 'Stay' column)
   
    # Split the data into training and testing sets using train_test_split()
    # splits the feature DataFrame and target variable Series into training and testing datasets.
    # 80 % of the data is used for training and 20 % for testing, with a random seed of 100 for reproducibility.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

    #Return the training and testing data splits
    return X_train, X_test, y_train, y_test

def model(train):
    # do not edit the predefined function name
    X_train, X_test, y_train, y_test = splitting(train)

    # Create the XGBoost classifier with specified hyperparameters
    classifier_xgb = xgboost.XGBClassifier(max_depth=4, learning_rate=0.1, n_estimators=800,
                                           objective='multi:softmax', reg_alpha=0.5, reg_lambda=1.5,
                                           booster='gbtree', n_jobs=4, min_child_weight=2, base_score=0.75)
    
    classifier_xgb.fit(X_train,y_train)

    # Make predictions on the x_test data
    y_pred = classifier_xgb.predict(X_test)

    # Calculate the accuracy score on the test data
    acc_score_xgb = accuracy_score(y_test, y_pred)

    # Calculate Precision
    precision = precision_score(y_test, y_pred, average='weighted')

    # Calculate Recall
    recall = recall_score(y_test, y_pred, average='weighted')

    # Calculate F1 Score
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Save the model to a file in the current directory
    with open('model.pkl', 'wb') as file:
        pickle.dump(classifier_xgb, file)

    # Return the accuracy score rounded to two decimal places
    return {'Accuracy':acc_score_xgb,'Precision':precision,'Recall':recall,'F1-score':f1}


def predict(test):
    # Load the model from the saved .pkl file
    with open('model.pkl', 'rb') as file:
        loaded_model = pickle.load(file)

    test1 = drop_duplicates_test(test)

    # Make predictions on the test data using the trained XGBoost classifier
    pred_xgb = loaded_model.predict(test1.iloc[:, 1:])

    # Create a DataFrame to store the predicted 'Stay' values
    # Add the 'case_id' column from the original test data to the result DataFrame
    # Rearrange the columns in the DataFrame to have 'case_id' as the first column and 'Stay' as the second column
    result_xgb = pd.DataFrame({'case_id': test1['case_id'], 'Stay': pred_xgb})

    # Map numeric predictions to their corresponding 'Stay' categories using the .replace() method
    # replace the following data in stay column with 0: '0-10', 1: '11-20', 2: '21-30', 3: '31-40', 4: '41-50', 5: '51-60', 6: '61-70', 7: '71-80', 8: '81-90', 9: '91-100', 10: 'More than 100 Days'
    result_xgb['Stay'] = result_xgb['Stay'].replace({
        0: '0-10', 1: '11-20', 2: '21-30', 3: '31-40', 4: '41-50',
        5: '51-60', 6: '61-70', 7: '71-80', 8: '81-90', 9: '91-100',
        10: 'More than 100 Days'
    })
    # Return the DataFrame with 'case_id' and predicted 'Stay' values
    return result_xgb

def result(test):
    # do not edit the predefined function name
    result_xgb = predict(test)

    # Group the data by 'Stay' and calculate the count of unique 'case_id' values in each group
    grouped_result = result_xgb.groupby('Stay')['case_id'].nunique().reset_index()

    # Rename the count column to 'Count'
    grouped_result.rename(columns={'case_id': 'Count'}, inplace=True)

    # Return the count of unique 'case_id' values for each 'Stay' category
    return grouped_result


train = separating_train()
test = separating_test()
train, test = get_countid_encode(train, test, ['patientid'], name = 'count_id_patient')
train, test = get_countid_encode(train, test,
                                 ['patientid', 'Hospital_region_code'], name = 'count_id_patient_hospitalCode')
train, test = get_countid_encode(train, test,
                                 ['patientid', 'Ward_Facility_Code'], name = 'count_id_patient_wardfacilityCode')
#print(result(train,test))

with open('model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Streamlit app title
st.title("XGBoost Model for Hospital Stay Duration Prediction")

# Upload the test dataset
test_data = st.file_uploader("Upload your test CSV file", type=['csv'])

if test_data:
    test = pd.read_csv(test_data)
    test['Bed Grade'].fillna(test['Bed Grade'].mode()[0], inplace=True)
    test['City_Code_Patient'].fillna(test['City_Code_Patient'].mode()[0], inplace=True)

    test['Stay'] = -1
    categorical_columns = ['Hospital_type_code', 'Hospital_region_code', 'Department', 'Ward_Type', 'Ward_Facility_Code', 'Type of Admission', 'Severity of Illness', 'Age']

    label_encoder = LabelEncoder()
    for column in categorical_columns:
        test[column] = label_encoder.fit_transform(test[column])

    test = get_countid_encode_test(test, ['patientid'], name = 'count_id_patient')
    test = get_countid_encode_test(test, ['patientid', 'Hospital_region_code'], name = 'count_id_patient_hospitalCode')
    test = get_countid_encode_test(test, ['patientid', 'Ward_Facility_Code'], name = 'count_id_patient_wardfacilityCode')

    columns_to_drop = ['Stay', 'patientid', 'Hospital_region_code', 'Ward_Facility_Code']
    test1 = test.drop(columns=columns_to_drop)

    pred_xgb = loaded_model.predict(test1.iloc[:, 1:])
    result_xgb = pd.DataFrame({'case_id': test1['case_id'], 'Stay': pred_xgb})

    result_xgb['Stay'] = result_xgb['Stay'].replace({
        0: '0-10', 1: '11-20', 2: '21-30', 3: '31-40', 4: '41-50',
        5: '51-60', 6: '61-70', 7: '71-80', 8: '81-90', 9: '91-100',
        10: 'More than 100 Days'
    })

    st.subheader('Predicted Hospital Stay Durations:')
    st.write(result_xgb)

    grouped_result = result_xgb.groupby('Stay')['case_id'].nunique().reset_index()

    grouped_result.rename(columns={'case_id': 'Count'}, inplace=True)

    st.subheader('Grouped Results:')
    st.write(grouped_result)

