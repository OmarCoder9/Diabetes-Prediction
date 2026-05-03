import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
plot_dir = "Project_plots"
os.makedirs(plot_dir, exist_ok=True)
# 1. load dataset
print("="*45)
print("Step #1: DataSet information")
print("="*45 + "\n")

df = pd.read_csv("./Data/Dataset_of_Diabetes.csv")
print("Dataset info")
df.info()
print("\n--- First 5 Rows ---")
print(df.head().to_string(index=False))


# 2. Cleaning Data
df['CLASS'] = df['CLASS'].str.strip()
df['Gender'] = df['Gender'].str.strip().str.upper()
df.drop_duplicates(inplace=True)
df_clean = df.drop(['ID', 'No_Pation'], axis=1)

print("\n" + "="*45)
print("Step #2: Missing Data and Cleaning")
print("="*45 + "\n")

plt.figure(figsize=(10,4))
sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.title("Missing Data Map (Yellow lines = Missing)")
plt.savefig(f"{plot_dir}/Missing_Data_Map.png", dpi=300, bbox_inches = 'tight')
plt.show()


has_null_series = df.isnull().any(axis=1)
missing_comparison = pd.crosstab(has_null_series, df['CLASS'])
print("Comparison of Missing Data vs Diabetes Class:\n")
print(missing_comparison.to_string())

# 3.Detecting Outliers
print("\n" + "="*45)
print("Step #3: Outlier Detection")
print("="*45 + "\n")

#Using boxplots
plt.figure(figsize=(18,12))
features_to_plot = ['Urea', 'Cr', 'HbA1c', 'Chol', 'TG', 'HDL', 'LDL', 'VLDL', 'BMI']
for i, col in enumerate(features_to_plot):
    plt.subplot(3,3, i + 1)
    sns.boxplot(y=df_clean[col], color='skyblue')
    plt.title(f"Outliers in {col}")
    plt.tight_layout()
plt.savefig(f"{plot_dir}/Outliers_Boxplots.png", dpi=300, bbox_inches = 'tight')
plt.show()

for col in features_to_plot:
    # 1. Calculate IQR Outliers
    Q1 = df_clean[col].quantile(0.25)
    Q3 = df_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    iqr_outliers = ((df_clean[col] < (Q1 - 1.5 * IQR)) | (df_clean[col] > (Q3 + 1.5 * IQR))).sum()

    # 2. Calculate Z-Score Outliers
    z_scores = np.abs(stats.zscore(df_clean[col]))
    z_outliers = (z_scores > 3).sum()

    print(f"{col:5} -> IQR Outliers found: {iqr_outliers:3} | Z-Score Outliers found: {z_outliers:3}")


#Handling Outliers
def handle_outliers_IQR(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[col] =  np.where(df[col] > upper_bound, upper_bound, np.where(df[col] < lower_bound, lower_bound, df[col]))
    return df

def handle_outliers_zScore(df, col):
    z_score = np.abs(stats.zscore(df[col]))
    mean = df[col].mean()
    std =  df[col].std()
    lower_bound = mean - (3 * std)
    upper_bound = mean + (3 * std)
    df[col] = np.where(z_score > 3, np.where(df[col] > mean, upper_bound ,lower_bound), df[col])
    return df


for col in ['Urea', 'Cr', 'VLDL']:
    df_clean = handle_outliers_IQR(df_clean, col)

# for col in ['Urea', 'Cr', 'VLDL']:
#     df_clean = handle_outliers_zScore(df_clean, col)


# Study and Observation 
print("\n" + "="*45)
print("Step #4: Study & Observation")
print("="*45 + "\n")
young_people = df_clean[df_clean['AGE'] < 40]
young_diabetic_percentage = (young_people['CLASS'] == 'Y').mean() * 100
print("Q1: What percentage of younger people are prone to be diagnosed with diabetes disease?")
print(f"Observation: percentage of people under 40 diagnosed as Diabetic: {young_diabetic_percentage:.2f}%")


print("\nQ2: Are women more prone to diabetes, or is it the other way?")
gender_dist = df_clean.groupby('Gender')['CLASS'].value_counts(normalize=True).unstack() * 100
gender_dist = gender_dist.map(lambda x: f"{x:.2f}%" if pd.notnull(x) else "0.00%")
print("Gender-wise Diabetes Distribution:")
print('-'*45)
print(gender_dist.to_string())



print("\n" + "="*45)
print("Step #5: Handling Imbalanced Data (SMOTE)")
print("="*45 + "\n")

label_encoder = LabelEncoder()
df_clean['Gender'] = label_encoder.fit_transform(df_clean['Gender'])
df_clean['CLASS'] = label_encoder.fit_transform(df_clean['CLASS'])

X = df_clean.drop('CLASS', axis=1)
y = df_clean['CLASS']



x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.fit_transform(x_test)

class_mapping = {0: "Non-Diabetic", 1:'Predicted-Diabetic', 2:'Diabetic'}

print("Class Distribution before SMOTE:")
before_smote = pd.Series(y_train).map(class_mapping).value_counts()
print(before_smote.to_string())

smote = SMOTE(random_state=42)
x_train_smote, y_train_smote = smote.fit_resample(x_train_scaled, y_train)

print("\nClass Distribution after SMOTE: ")
after_smote = pd.Series(y_train_smote).map(class_mapping).value_counts()
print(after_smote.to_string())


print("\n" + "="*45)
print("Step #6: Machine Learning Models")
print("="*45 + "\n")

models = {
    "Decision Tree": DecisionTreeClassifier(criterion="entropy", min_samples_leaf=10, random_state=42), 
    "Naive Bayes": GaussianNB(),
    "Logistec Regression": LogisticRegression(max_iter=1000, random_state=42)
}

k_values = [4, 5, 6, 7, 9, 10]
for k in k_values:
    models[f"KNN (K = {k})"] = KNeighborsClassifier(n_neighbors=k)

results = {}

print(f"{'Algorithm':<20} | {'Accuracy(%)':<10}")
print("-"*45)

for name, model in models.items():
    model.fit(x_train_smote, y_train_smote)
    y_pred = model.predict(x_test_scaled)
    acc = accuracy_score(y_test, y_pred) * 100
    results[name] = acc
    print(f"{name:<20} | {acc:.2f}%")

print("-" * 45)

best_model_name = max(results, key=results.get)
best_model_score = results[best_model_name]
print(f"The Best Overall Algorithm: {best_model_name} with {best_model_score:.2f}% Accuracy\n")


#Visualization and Results
#Plot Accuracy Comparision
plt.figure(figsize=(14,5))
sns.barplot(x=list(results.keys()), y=list(results.values()),hue=list(results.keys()), palette="viridis", legend=False)
plt.ylabel('Accuracy Score (%)')
plt.title("Algorithm Accuracy Comparison")
plt.ylim(0,100)
plt.savefig(f"{plot_dir}/Algorithm_Accuracy.png", dpi=300, bbox_inches = 'tight')
plt.show()

#plotting the Decision Tree
plt.figure(figsize=(20, 10))
tree_model = models["Decision Tree"]
plot_tree(tree_model, feature_names=X.columns, class_names=list(label_encoder.classes_), filled=True, rounded=True, fontsize=6)
plt.title("Decision Tree")
plt.savefig(f"{plot_dir}/Decision_Tree.png", dpi=300, bbox_inches='tight')
plt.show()


#Correlation Heatmap
plt.figure(figsize=(12,8))
sns.heatmap(df_clean.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Matrix of Diabetes Factors")
plt.savefig(f"{plot_dir}/Correlation_Matrix.png", dpi=300, bbox_inches = 'tight')
plt.show()

# Age Distribution by Class
plt.figure(figsize=(10,6))
sns.histplot(data=df, x='AGE', hue='CLASS', multiple="stack")
plt.title("Age Distribution by Diabetes Class")
plt.savefig(f"{plot_dir}/Age_Distribution.png", dpi=300, bbox_inches = 'tight')
plt.show()


#Confusion Matrix for the best model
winning_model = models[best_model_name]
y_pred_best = winning_model.predict(x_test_scaled)
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=list(class_mapping.values()),yticklabels=list(class_mapping.values()))
plt.title(f"Confusion Matrix: {best_model_name}")
plt.ylabel("Actual Diagnosis")
plt.xlabel("Predicted Diagnosis")
plt.savefig(f"{plot_dir}/Confusion_Matrix.png", dpi=300, bbox_inches = 'tight')
plt.show()