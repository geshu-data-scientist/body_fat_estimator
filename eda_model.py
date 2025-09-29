#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn as sns
import scipy.stats as stats
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


# In[2]:


data=pd.read_csv('bodyfat.csv')


# In[3]:


data.shape


# In[4]:


data.info()


# In[5]:


data.describe()


# In[6]:


data.head()


# In[7]:


num_columns = len(data.columns)
num_rows = (num_columns // 4) + 1  # Assuming 4 columns per row

# Set up subplots
fig, axes = plt.subplots(num_rows, 4, figsize=(15, 10))
axes = axes.flatten()

# Iterate over each column and plot histogram
for i, column in enumerate(data.columns):
    ax = axes[i]
    ax.hist(data[column], bins=20, color='skyblue', edgecolor='black')
    ax.set_title(column)
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')


# In[8]:


# Display correlation matrix
corr_matrix = data.corr()

# Create a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()


# In[11]:


numeric_cols = data.select_dtypes(include=["number"]).columns

for col in numeric_cols:
    plt.figure(figsize=(15, 4))

    # 1. Box Plot
    plt.subplot(1, 3, 1)
    sns.boxplot(y=data[col], color="skyblue")
    plt.title(f"Box Plot of {col}")

    # 2. Gaussian Distribution (Histogram + KDE)
    plt.subplot(1, 3, 2)
    sns.histplot(data[col], kde=True, stat="density", bins=30, color="lightcoral")
    # Plot Gaussian curve with mean & std
    mu, sigma = data[col].mean(), data[col].std()
    x = np.linspace(data[col].min(), data[col].max(), 100)
    plt.plot(x, stats.norm.pdf(x, mu, sigma), "k", linewidth=2, label="Normal PDF")
    plt.title(f"Distribution of {col}\nμ={mu:.2f}, σ={sigma:.2f}")
    plt.legend()

    # 3. QQ Plot
    plt.subplot(1, 3, 3)
    stats.probplot(data[col].dropna(), dist="norm", plot=plt)
    plt.title(f"QQ Plot of {col}")

    plt.tight_layout()
    plt.show()


# In[12]:


import pandas as pd

def detect_outliers_iqr(data, columns=None):
    """
    Detect outliers in a DataFrame using the IQR method.

    Parameters:
        data (pd.DataFrame): Input dataframe
        columns (list, optional): List of columns to check. If None, checks all numeric columns.

    Returns:
        dict: {column_name: list_of_outlier_indices}
    """
    if columns is None:
        columns = data.select_dtypes(include=["number"]).columns

    outliers = {}
    for col in columns:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        # Get indices of outliers
        outlier_idx = data[(data[col] < lower) | (data[col] > upper)].index.tolist()
        outliers[col] = outlier_idx

    return outliers
outlier_dict = detect_outliers_iqr(data)

outlier_dict


# In[13]:


import pandas as pd

def remove_outliers_iqr(data, columns=None):
    """
    Remove rows containing outliers (IQR method) from a DataFrame.

    Parameters:
        data (pd.DataFrame): Input dataframe
        columns (list, optional): Columns to check. If None, checks all numeric columns.

    Returns:
        pd.DataFrame: Cleaned dataframe with outliers removed
    """
    if columns is None:
        columns = data.select_dtypes(include=["number"]).columns

    cleaned_data = data.copy()
    for col in columns:
        Q1 = cleaned_data[col].quantile(0.25)
        Q3 = cleaned_data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        # Keep only rows within bounds
        cleaned_data = cleaned_data[(cleaned_data[col] >= lower) & (cleaned_data[col] <= upper)]

    return cleaned_data

cleaned_data = remove_outliers_iqr(data)
print(f"Original shape: {data.shape}, After outlier removal: {cleaned_data.shape}")


# In[14]:


data.columns


# In[15]:


X=data.drop(['BodyFat'],axis=1)
y=data['BodyFat']


# In[16]:


import pandas as pd

def correlation_filter(data, threshold=0.8):
    """
    Remove features that are highly correlated (above threshold).

    Parameters:
        data (pd.DataFrame): Input dataframe (numeric features only recommended)
        threshold (float): Correlation threshold for dropping features (0 < threshold < 1)

    Returns:
        pd.DataFrame: DataFrame with reduced features
        list: List of dropped columns
    """
    corr_matrix = data.corr().abs()   # absolute correlation
    upper = corr_matrix.where(
        pd.np.triu(pd.np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    # Find features to drop
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    reduced_data = data.drop(columns=to_drop)
    return reduced_data, to_drop

# Example usage:
reduced_data, dropped = correlation_filter(X, threshold=0.85)
print("Dropped columns:", dropped)


# In[17]:


X.shape,y.shape


# In[18]:


reduced_data.shape


# In[19]:


reduced_data.columns


# In[20]:


y.shape


# In[21]:


def mutual_info_scores(X, y, problem_type="classification"):
    """
    Calculate mutual information between X and y.

    Parameters:
        X (pd.DataFrame or np.array): Feature matrix
        y (pd.Series or np.array): Target variable
        problem_type (str): "classification" or "regression"

    Returns:
        pd.DataFrame: Features and their MI scores sorted descending
    """
    if problem_type == "classification":
        mi_scores = mutual_info_classif(X, y, random_state=42)
    else:
        mi_scores = mutual_info_regression(X, y, random_state=42)

    mi_df = pd.DataFrame({
        "Feature": X.columns if isinstance(X, pd.DataFrame) else [f"X{i}" for i in range(X.shape[1])],
        "Mutual Information": mi_scores
    }).sort_values(by="Mutual Information", ascending=False).reset_index(drop=True)

    return mi_df

# Example usage:
mi_df = mutual_info_scores(X, y, problem_type="regression")


# In[22]:


list(mi_df[0:5]['Feature'])


# In[23]:


# selecting these =['Density', 'Abdomen', 'Chest', 'Hip', 'Weight']


# In[24]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.feature_selection import SelectFromModel

def tree_based_regression_feature_selection(X, y, model_type="random_forest", threshold="median", plot=True):
    """
    Select features for regression using tree-based models.

    Parameters:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target variable
        model_type (str): "random_forest", "xgboost", "lightgbm", "catboost"
        threshold (str or float): Threshold to select features ("median", "mean", or numeric)
        plot (bool): Whether to plot feature importances

    Returns:
        pd.DataFrame: Reduced feature matrix with selected features
        list: Selected feature names
        pd.DataFrame: Feature importance scores
    """
    # Initialize model
    if model_type == "random_forest":
        model = RandomForestRegressor(n_estimators=200, random_state=42)
    elif model_type == "xgboost":
        model = XGBRegressor(n_estimators=200, random_state=42, verbosity=0)
    elif model_type == "lightgbm":
        model = LGBMRegressor(n_estimators=200, random_state=42)
    elif model_type == "catboost":
        model = CatBoostRegressor(n_estimators=200, random_state=42, verbose=0)
    else:
        raise ValueError("model_type must be 'random_forest', 'xgboost', 'lightgbm', or 'catboost'")

    # Fit model
    model.fit(X, y)

    # Feature importance
    feature_importances = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

    # Select features above threshold
    selector = SelectFromModel(model, threshold=threshold, prefit=True)
    selected_features = X.columns[selector.get_support()].tolist()
    X_reduced = X[selected_features]

    # Optional plot
    if plot:
        plt.figure(figsize=(10,6))
        plt.barh(feature_importances['Feature'], feature_importances['Importance'], color='skyblue')
        plt.gca().invert_yaxis()
        plt.title(f"{model_type.title()} Feature Importances")
        plt.show()

    return X_reduced, selected_features, feature_importances
X_selected, selected_features, feature_importances = tree_based_regression_feature_selection(X, y, model_type="random_forest", threshold="median")
print("Selected features:", selected_features)


# In[25]:


model_types = [
    "random_forest",      # sklearn.ensemble.RandomForestRegressor
    "xgboost",            # xgboost.XGBRegressor
    "catboost"]         # lightgbm.LGBMRegressor
    


# In[26]:




def tree_based_regression_feature_selection(X, y, model_type=model_types[0], threshold="median", plot=True):
    """
    Select features for regression using tree-based models.

    Parameters:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target variable
        model_type (str): "random_forest", "xgboost", "lightgbm", "catboost"
        threshold (str or float): Threshold to select features ("median", "mean", or numeric)
        plot (bool): Whether to plot feature importances

    Returns:
        pd.DataFrame: Reduced feature matrix with selected features
        list: Selected feature names
        pd.DataFrame: Feature importance scores
    """
    # Initialize model
    if model_type == "random_forest":
        model = RandomForestRegressor(n_estimators=200, random_state=42)
    elif model_type == "xgboost":
        model = XGBRegressor(n_estimators=200, random_state=42, verbosity=0)
    elif model_type == "lightgbm":
        model = LGBMRegressor(n_estimators=200, random_state=42)
    elif model_type == "catboost":
        model = CatBoostRegressor(n_estimators=200, random_state=42, verbose=0)
    else:
        raise ValueError("model_type must be 'random_forest', 'xgboost', 'lightgbm', or 'catboost'")

    # Fit model
    model.fit(X, y)

    # Feature importance
    feature_importances = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

    # Select features above threshold
    selector = SelectFromModel(model, threshold=threshold, prefit=True)
    selected_features = X.columns[selector.get_support()].tolist()
    X_reduced = X[selected_features]

    # Optional plot
    if plot:
        plt.figure(figsize=(10,6))
        plt.barh(feature_importances['Feature'], feature_importances['Importance'], color='skyblue')
        plt.gca().invert_yaxis()
        plt.title(f"{model_type.title()} Feature Importances")
        plt.show()

    return X_reduced, selected_features, feature_importances


# In[27]:


X_selected, selected_features, feature_importances = tree_based_regression_feature_selection(X, y, model_type=model_types[0], threshold="median")
print("Selected features:", selected_features)


# In[28]:


X_selected, selected_features, feature_importances = tree_based_regression_feature_selection(X, y, model_type=model_types[1], threshold="median")
print("Selected features:", selected_features)


# In[29]:


X_selected, selected_features, feature_importances = tree_based_regression_feature_selection(X, y, model_type=model_types[2], threshold="median")
print("Selected features:", selected_features)


# In[30]:


#we see comomn feature are 

random_forest=['Density', 'Height', 'Chest', 'Abdomen', 'Hip', 'Thigh', 'Forearm']
xg_boost=['Density', 'Weight', 'Height', 'Abdomen', 'Thigh', 'Ankle', 'Wrist']
catbpost=['Density', 'Weight', 'Height', 'Chest', 'Abdomen', 'Knee', 'Wrist']
mutiula_information=['Density', 'Abdomen', 'Chest', 'Hip', 'Weight']
pearson_corrrelation_08=['Density', 'Age', 'Weight', 'Height', 'Neck', 'Ankle', 'Biceps',
       'Forearm', 'Wrist']


# In[31]:


common=['Density','Height','Abdomen','Forearm']


# In[32]:


X1=X[common]
y=y


# In[33]:


#training random forst


# In[34]:





# In[35]:


X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size=0.2, random_state=42)
final_model = RandomForestRegressor(n_estimators=200, random_state=42)
final_model.fit(X_train, y_train)


# In[40]:


import shap
import matplotlib.pyplot as plt

# model = trained tree-based regression model (RandomForest/XGBoost/etc.)
# X = dataframe of selected features after filter/wrapper methods

explainer = shap.Explainer(final_model, X_train)   # TreeExplainer works automatically for tree models
shap_values = explainer(X_train)

# Summary plot: shows global feature importance
shap.summary_plot(shap_values, X_train)

# Bar plot for overall feature importance
shap.summary_plot(shap_values, X_train, plot_type="bar")


# In[42]:


#based on this shap interpretation i reomve forearm as it is mostly around 0


# In[45]:


common.remove('Forearm')


# In[46]:


common


# In[47]:


X_train, X_test, y_train, y_test = train_test_split(X1[common], y, test_size=0.2, random_state=42)
final_model = RandomForestRegressor(n_estimators=200, random_state=42)
final_model.fit(X_train, y_train)


# Metric	Meaning	Ideal
# R² Score	Proportion of variance explained	Closer to 1
# MSE	Squared error	Smaller better
# RMSE	Root of MSE, same units as target	Smaller better
# MAE	Average absolute error	Smaller better

# In[48]:


y_train_pred = final_model.predict(X_train)
y_test_pred = final_model.predict(X_test)


# In[49]:


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt


# In[50]:


def regression_report(y_true, y_pred, dataset_name="Dataset"):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"--- {dataset_name} Regression Report ---")
    print(f"R² Score       : {r2:.4f}")
    print(f"Mean Squared Error (MSE) : {mse:.4f}")
    print(f"Root MSE       : {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print("\n")
    
# Train metrics
regression_report(y_train, y_train_pred, "Train")

# Test metrics
regression_report(y_test, y_test_pred, "Test")


# In[52]:


plt.figure(figsize=(6,6))
plt.scatter(y_test, y_test_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.title("Random Forest Regression: True vs Predicted")
plt.show()
#The red dashed line is the ideal prediction (y = y_pred).

#Points close to the line indicate good predictions.


# In[53]:


from sklearn.ensemble import RandomForestRegressor

param_grid = {
    "n_estimators": [100, 200, 300, 400, 500],
    "max_depth": [None, 5, 10, 15, 20, 25],
    "min_samples_split": [2, 5, 10, 15],
    "min_samples_leaf": [1, 2, 4, 6],
    "max_features": ["auto", "sqrt", "log2"]
}

rf = RandomForestRegressor(random_state=42)


# In[57]:


from sklearn.model_selection import RandomizedSearchCV

# Randomized search
random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_grid,
    n_iter=50,          # number of random combinations to try
    scoring='r2',
    cv=5,               # 5-fold cross-validation
    verbose=2,
    random_state=42,
    n_jobs=-1           # use all processors
)

# Fit
random_search.fit(X_train, y_train)

# Best hyperparameters
print("Best Parameters:", random_search.best_params_)

# Best estimator
best_rf = random_search.best_estimator_


# In[60]:


best_rf = random_search.best_estimator_


# In[61]:


from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np

# Predictions
y_train_pred = best_rf.predict(X_train)
y_test_pred = best_rf.predict(X_test)

# Function to print regression metrics
def regression_report(y_true, y_pred, dataset_name="Dataset"):
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    
    print(f"--- {dataset_name} Regression Report ---")
    print(f"R² Score       : {r2:.4f}")
    print(f"MSE            : {mse:.4f}")
    print(f"RMSE           : {rmse:.4f}")
    print(f"MAE            : {mae:.4f}")
    print("\n")

# Train report
regression_report(y_train, y_train_pred, "Train")

# Test report
regression_report(y_test, y_test_pred, "Test")


# In[62]:


import joblib

# Save the trained model
joblib.dump(best_rf, "bodyfat_rf_model.pkl")
print("Model saved successfully!")

# Load the model later
loaded_model = joblib.load("bodyfat_rf_model.pkl")
y_test_pred = loaded_model.predict(X_test)  # Use loaded model for predictions


# In[63]:


import pickle

# Save feature columns
with open("bodyfat_features.pkl", "wb") as f:
    pickle.dump(list(X_train.columns), f)

# Later load the columns
with open("bodyfat_features.pkl", "rb") as f:
    feature_columns = pickle.load(f)

print("Model expects these features:", feature_columns)


# In[ ]:




