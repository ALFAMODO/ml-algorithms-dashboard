import streamlit as st  
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder

import pickle

from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb

#---------------------------------#
# Page layout
## Page expands to full width
st.set_page_config(page_title='The Machine Learning App',
    layout='wide')

#---------------------------------#


def create_bar_chart(df, x, y):
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x=x, y=y)
    plt.title(f'Bar Chart of {y} vs {x}')
    st.pyplot(plt)

# Function to create a line chart
def create_line_chart(df, x, y):
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x=x, y=y)
    plt.title(f'Line Chart of {y} vs {x}')
    st.pyplot(plt)

# Function to create a pie chart
def create_pie_chart(df, column):
    plt.figure(figsize=(8, 8))
    df[column].value_counts().plot.pie(autopct='%1.1f%%')
    plt.title(f'Pie Chart of {column}')
    st.pyplot(plt)

# Function to create a boxplot
def create_boxplot(df, x, y):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x=x, y=y)
    plt.title(f'Boxplot of {y} by {x}')
    st.pyplot(plt)


def make_histogram(df, target_feature, bins = 10, custom_ticks=None, unit='', additional=''):
    plt.figure(figsize=(10, 5))
    plt.hist(df[target_feature], bins=bins)
    if custom_ticks is not None:
        plt.xticks(custom_ticks)
    plt.ylabel('Count')
    plt.xlabel(target_feature)
    plt.title(f"Distribution of {target_feature.lower()}{additional}:\n")
    plt.grid()
    plt.show()
    print(f"Distribution of {target_feature.lower()}{additional}: {df[target_feature].mean():.2f} ± {df[target_feature].median():.2f} {unit}\nMedian: {df[target_feature].median():.2f} {unit}\nMinimum: {df[target_feature].min()} {unit}\nMaximum: {df[target_feature].max()} {unit}\n{df[target_feature].skew():.3f} Skewness\n")

def make_piechart(df, target_feature, additional=''):
    dict_of_val_counts = dict(df[target_feature].value_counts())
    data = list(dict_of_val_counts.values())
    keys = list(dict_of_val_counts.keys())
    
    palette_color = sns.color_palette('bright')
    plt.pie(data, labels=keys, colors=palette_color, autopct='%.0f%%')
    plt.title(f"Distribution of Cutomer's {target_feature}:")
    plt.show()
    print_str = f"Distribution of cutomer's {target_feature.lower()}{additional}:"
    for k, v in zip(keys, data):
        print_str += f"\n{v} {k}"
    print(print_str)

def make_barplot(df, target_feature, custom_ticks=None, unit='', additional=''):
    plt.figure(figsize=(10, 5))
    dict_of_val_counts = dict(df[target_feature].value_counts())
    data = list(dict_of_val_counts.values())
    keys = list(dict_of_val_counts.keys())
    plt.bar(keys, data)
    if custom_ticks is not None:
        plt.xticks(custom_ticks)
    plt.xlabel(f'{target_feature.capitalize()}{additional}')
    plt.ylabel('Frequency')
    plt.title(f"Distribution of customer's {str(target_feature).lower()}{additional}\n")  # Convert target_feature to string
    plt.grid(axis='y')
    st.pyplot(plt)
    st.write(f"Distribution of customer's {str(target_feature).lower()}{additional}: {df[target_feature].mean():.2f} ± {df[target_feature].median():.2f} {unit}\nMedian: {df[target_feature].median():.2f} {unit}\nMinimum: {df[target_feature].min()} {unit}\nMaximum: {df[target_feature].max()} {unit}\n\n Skewness: {df[target_feature].skew():.3f} \n")


def make_boxplot(df, feature):
    plt.figure(figsize=(10,5))
    sns.boxplot(df, x=feature)
    plt.title(f"Boxplot of {feature}\n")
    plt.xlabel(feature)
    plt.ylabel("Values")
    plt.show()

def categorize_tenure(tenure):
    if tenure < 12:
        return '0-1 year'
    elif 12 <= tenure < 24:
        return '1-2 years'
    elif 24 <= tenure < 36:
        return '2-3 years'
    elif 36 <= tenure < 48:
        return '3-4 years'
    elif 48 <= tenure < 60:
        return '4-5 years'
    else:
        return '5+ years'

def data_process(data):
    data.columns = [col.lower().replace(' ', '_') for col in data.columns]
    missing_data = data[data.isnull().any(axis=1)]

    if missing_data.empty:
        st.write("✓ No Missing Values Found")
    else:
        # Drop rows with null values
        data = data.dropna()
        st.write("✓ Missing values found and dropped.")

    duplicate_rows = data[data.duplicated()]

    if duplicate_rows.empty:
        st.write("✓ No Duplicate Values Found")
    else:
        data = data.drop_duplicates(keep='first')
        st.write("✓Duplicate values found and dropped")

    # Check for duplicate CustomerID
    if 'customerid' in data.columns:
        if data['customerid'].duplicated().any():
            st.write("Duplicate Customer IDs found. Keeping the Customer ID column.")
        else:
            st.write("No duplicate Customer IDs found. Removing the Customer ID column.")
            data = data.drop(columns=['customerid'])

    for col in data.select_dtypes(include=['float', 'int']).columns:
        # Convert numeric columns in a DataFrame to integer type where possible.
        # Check if column can be converted to integers without losing information
        if (data[col] % 1 == 0).all():  # Check if all values are whole numbers
            data[col] = data[col].astype(int)
            print(f"Converted column '{col}' to integer type.")
        else:
            print(f"Column '{col}' contains non-whole numbers and cannot be safely converted to integer type.")
    
    st.write(data.head())

def data_visualization(df):
    

    #####################################################

    gender_churn = df.groupby(['gender', 'churn']).size().unstack()

    X = list(gender_churn.index)
    churn_0 = list(gender_churn.iloc[:, 0])
    churn_1 = list(gender_churn.iloc[:, 1])

    X_axis = np.arange(len(X))

    plt.figure(figsize=(10, 6))
    plt.bar(X_axis - 0.2, churn_1, 0.4, label='Churn')
    plt.bar(X_axis + 0.2, churn_0, 0.4, label='Not Churn')

    plt.xticks(X_axis, X)
    plt.xlabel('Gender')
    plt.ylabel('Count')
    plt.title("Gender-wise Churn Rate")
    plt.legend(loc='center right')
    plt.grid(axis='y')
    st.pyplot(plt)

    ##############################################

    filtered = df.groupby(['payment_delay', 'churn']).size().unstack()

    X = list(filtered.index)
    churn_0 = list(filtered.iloc[:, 0])
    churn_1 = list(filtered.iloc[:, 1])
    
    X_axis = np.arange(len(X))
    
    plt.figure(figsize=(12, 6))
    plt.bar(X_axis - 0.2, churn_1, 0.4, label='Churn')
    plt.bar(X_axis + 0.2, churn_0, 0.4, label='Not Churn')
    
    plt.xticks(X_axis, X, rotation=90)
    plt.xlabel("Customer payment delays in days")
    plt.ylabel('Count')
    plt.title("Churn rate based on payment delays")
    plt.legend(loc='center right')
    plt.grid(axis='y')
    
    st.pyplot(plt)
    
    ##################################################

    filtered = df.copy()
    filtered['tenure_segmentation'] = df['tenure'].apply(categorize_tenure)
    filtered = filtered.groupby(['tenure_segmentation', 'churn']).size().unstack()

    X = list(filtered.index)
    churn_0 = list(filtered.iloc[:, 0])
    churn_1 = list(filtered.iloc[:, 1])
    
    X_axis = np.arange(len(X))
    
    plt.figure(figsize=(12, 6))
    plt.bar(X_axis - 0.2, churn_1, 0.4, label='Churn')
    plt.bar(X_axis + 0.2, churn_0, 0.4, label='Not Churn')
    
    plt.xticks(X_axis, X, rotation=45)
    plt.xlabel('Tenures')
    plt.ylabel('Count')
    plt.title("Churn rate based on tenures")
    plt.legend(loc='center right')
    plt.grid(axis='y')
    
    st.pyplot(plt)

    ####################################################

    st.subheader("Interactive Data Visualization")

    # Allow user to choose the type of chart
    chart_type = st.selectbox("Select Chart Type", ["Bar Chart", "Line Chart", "Pie Chart", "Boxplot"])

    # Allow user to select features for the chart
    if chart_type == "Pie Chart":
        column = st.selectbox("Select Feature for Pie Chart", df.columns)
        if st.button("Generate Pie Chart"):
            create_pie_chart(df, column)
    else:
        x = st.selectbox("Select X-axis Feature", df.columns)
        y = st.selectbox("Select Y-axis Feature", df.columns)
        if st.button(f"Generate {chart_type}"):
            if chart_type == "Bar Chart":
                create_bar_chart(df, x, y)
            elif chart_type == "Line Chart":
                create_line_chart(df, x, y)
            elif chart_type == "Boxplot":
                create_boxplot(df, x, y)

    st.subheader('Feature Selection for Barplot')
    
    features = df.columns.tolist()
    selected_feature = st.selectbox('Select a feature to create a barplot:', features)

    # Create barplot based on selected feature
    if selected_feature == 'Age':
        make_barplot(df, selected_feature, custom_ticks=np.arange(df[selected_feature].min(), df[selected_feature].max() + 1, 5))
    elif selected_feature == 'Tenure':
        make_barplot(df, selected_feature, custom_ticks=np.arange(0, 61, 3), additional=' (months)', unit='months')
    else:
        make_barplot(df, selected_feature)


def build_model(data):
    st.subheader('Data Modeling')
    
    # Check for duplicate customers
    duplicate_customers = data['Customer ID'].duplicated()
    
    if not duplicate_customers.any():
        X = data.drop(columns=['Transaction ID', 'Total Amount', 'Customer ID'])  # Exclude 'Total Amount' and 'Customer ID' from features
    else:
        X = data.drop(columns=['Transaction ID', 'Total Amount'])  # Exclude 'Total Amount' from features

    y = data['Total Amount']

    # Encode categorical variables
    label_encoder = LabelEncoder()
    X['Gender'] = label_encoder.fit_transform(X['Gender'])
    X['Product Category'] = label_encoder.fit_transform(X['Product Category'])

    # Calculate the correlation matrix (excluding 'Customer ID' from data)
    correlation_matrix = X.corr()

    # Create and configure the heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    ax.set_title('Correlation Heatmap')

    # Show plot in Streamlit
    st.pyplot(fig)

    # Split the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the linear regression model
    model = LinearRegression()

    # Train the model on the training data
    model.fit(X_train, y_train)

    st.subheader('Results')

    col1, col2 = st.columns(2)

    with col1:

        # Print model coefficients

        st.write("Intercept:", model.intercept_)
        st.write("Coefficients:", model.coef_)

        # Predict Total Amount using the trained model
        y_pred = model.predict(X_test)

        # Calculate Mean Squared Error (MSE) and R-squared
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Print evaluation metrics
        st.write("Mean Squared Error (MSE):", mse)
        st.write("R-squared (R2):", r2)

    with col2:

        # Create and configure the scatter plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(y_test, y_pred, color='blue', alpha=0.5)
        ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
        ax.set_title('Actual vs. Predicted Values')
        ax.set_xlabel('Actual Total Amount')
        ax.set_ylabel('Predicted Total Amount')

        # Show plot in Streamlit
        st.pyplot(fig)

def conclusion():
    st.subheader('Conclusion')
    st.write("The Mean Squared Error (MSE) measures the average squared difference between the predicted values and the actual values. In this case, the MSE value of approximately 42787.32 indicates that, on average, the squared difference between the predicted Total Amount values and the actual Total Amount values is around 42787.32. Lower values of MSE indicate better model performance, so this MSE value suggests that the model's predictions are relatively close to the actual values, but there is still room for improvement.")
    st.write("The R-squared (R2) score measures the proportion of the variance in the target variable (Total Amount) that is predictable from the features. The R2 score of approximately 0.8538 indicates that around 85.38% of the variance in the Total Amount can be explained by the features included in the model. Higher values of R2 indicate better model performance, so this R2 score suggests that the model is doing a good job of capturing the variation in Total Amount based on the features.")

def app():
    st.title('Customer Retention Analysis')
    #if st.button('Press to use Example Dataset'):
    data = pd.concat(
    [
        pd.read_csv('data_files/logistic data/train.csv'),
        pd.read_csv('data_files/logistic data/test.csv'),
    ], 
    axis=0)
    data.reset_index(drop=True, inplace=True)
    data_process(data)
    data_visualization(data)
    


if __name__ == "__main__":
    app()
