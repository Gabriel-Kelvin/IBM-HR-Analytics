import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np

# Read the data
df = pd.read_csv("Attrition.csv")

num_rows, num_cols = df.shape
print("Number of rows:", num_rows)
print("Number of columns:", num_cols)

missing_df = df.isnull().sum().to_frame().rename(columns={0:"Total No. of Missing Values"})
missing_df["% of Missing Values"] = round((missing_df["Total No. of Missing Values"]/len(df))*100,2)
missing_df

print("Inference:\nNone of the Attributes are having Missing Values.\nSince there's no missing values our further analysis will be consistent and unbaised.")

print(df.describe().T)

print("Inference:\nThe Minimum Age is 18 which conveys that All employees are Adult. So there's no need of Over18 Attribute for our analysis.\nThe Stanard Deviation value of EmployeeCount and StandardHours is 0.00 which conveys that All values present in this attribute are same.\nAttribute EmployeeNumber represents a unique value to each of the employees, which will not provide any meaningful inisghts.\nSince this Attribute will not provide any meaningful insights in our analysis we can simply drop these attributes.")

# Drop unnecessary columns
df.drop(['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours'], axis="columns", inplace=True)

print(df.head())

# Encoding categorical variables
categorical_cols = ['Attrition', 'BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus']

onehot_encoder = OneHotEncoder(sparse=False)
encoded_cols = pd.DataFrame(onehot_encoder.fit_transform(df[categorical_cols]))
feature_indices = onehot_encoder.get_feature_names_out(categorical_cols)
new_columns = [col.split('_')[0] for col in feature_indices]
encoded_cols.columns = new_columns
df = df.drop(categorical_cols, axis=1)
df_encoded = pd.concat([df, encoded_cols], axis=1)

print("DataFrame after encoding categorical variables:")
print(df_encoded.head())

# Removing duplicate columns
encoded_df = df_encoded.loc[:,~df_encoded.columns.duplicated()]

print("DataFrame after removing duplicate columns:")
print(encoded_df.head())

# Splitting the data
target_column = 'Attrition'
features = encoded_df.drop(columns=[target_column])
target = encoded_df[target_column]

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Logistic Regression
categorical_cols = features.select_dtypes(include=['object']).columns

categorical_transformer = Pipeline(steps=[
 ('imputer', SimpleImputer(strategy='most_frequent')),
 ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
 transformers=[('cat', categorical_transformer, categorical_cols)],
 remainder='passthrough')

logistic_model = Pipeline(steps=[
 ('preprocessor', preprocessor),
 ('classifier', LogisticRegression(random_state=42))])

logistic_model.fit(features, target)
logistic_predictions = logistic_model.predict(X_test)

logistic_accuracy = accuracy_score(y_test, logistic_predictions)
logistic_precision = precision_score(y_test, logistic_predictions)
logistic_recall = recall_score(y_test, logistic_predictions)
logistic_f1 = f1_score(y_test, logistic_predictions)

print("Logistic Regression Metrics:")
print("Accuracy:", logistic_accuracy)
print("Precision:", logistic_precision)
print("Recall:", logistic_recall)
print("F1 Score:", logistic_f1)

# Random Forest
categorical_cols = features.select_dtypes(include=['object']).columns

categorical_transformer = Pipeline(steps=[
 ('imputer', SimpleImputer(strategy='most_frequent')),
 ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
 transformers=[('cat', categorical_transformer, categorical_cols)],
 remainder='passthrough')

rf_model = Pipeline(steps=[
 ('preprocessor', preprocessor),
 ('classifier', RandomForestClassifier(random_state=42))])

rf_model.fit(features, target)
rf_predictions = rf_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_predictions)
rf_precision = precision_score(y_test, rf_predictions)
rf_recall = recall_score(y_test, rf_predictions)
rf_f1 = f1_score(y_test, rf_predictions)

print("\nRandom Forest Metrics:")
print("Accuracy:", rf_accuracy)
print("Precision:", rf_precision)
print("Recall:", rf_recall)
print("F1 Score:", rf_f1)

# Plotting metrics
models = ['Logistic Regression', 'Random Forest']
accuracy = [logistic_accuracy, rf_accuracy]
precision = [logistic_precision, rf_precision]
recall = [logistic_recall, rf_recall]
f1 = [logistic_f1, rf_f1]

plt.figure(figsize=(10, 5))
plt.bar(models, accuracy, color=['blue', 'green'])
plt.title('Accuracy of Models')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.show()

x = np.arange(len(models))
width = 0.2

fig, ax = plt.subplots(figsize=(10, 5))
rects1 = ax.bar(x - width, precision, width, label='Precision', color='blue')
rects2 = ax.bar(x, recall, width, label='Recall', color='green')
rects3 = ax.bar(x + width, f1, width, label='F1-score', color='orange')

ax.set_ylabel('Scores')
ax.set_title('Model Metrics')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()

plt.show()