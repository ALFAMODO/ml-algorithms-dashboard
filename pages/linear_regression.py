import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

#---------------------------------#
# Page layout
## Page expands to full width
st.set_page_config(page_title='The Machine Learning App',
    layout='wide')

#---------------------------------#
# Model building
def data_process(data):
    missing_data = data[data.isnull().any(axis=1)]

    if missing_data.empty:
        st.write("✓ No Missing Values Found")
    else:
        st.write('Missing Values Found')
        # Drop rows with null values
        data = data.dropna()
        st.write("Missing values have been removed.")

    duplicated_data = data.duplicated()
    duplicated_values = duplicated_data.sum()
    if duplicated_values:
        st.write(duplicated_values, 'Duplicate Values Found')
    else:
        st.write('✓ No Duplicate Values Found')

    numerical_variables = data.select_dtypes(include=['number']).columns.tolist()
    for var in numerical_variables:
        z_scores = np.abs((data[var] - data[var].mean()) / data[var].std())
        z_threshold = 3

        outliers_zscore = data[z_scores > z_threshold]

        Q1 = data[var].quantile(0.25)
        Q3 = data[var].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_threshold = Q1 - 1.5 * IQR
        upper_threshold = Q3 + 1.5 * IQR
        outliers_iqr = data[(data[var] < lower_threshold) | (data[var] > upper_threshold)]

        # Check if outliers are found using both methods for the current variable
        if not outliers_zscore.empty or not outliers_iqr.empty:
            st.write(f"Outliers found for '{var}' variable using Z-score method:")
            st.write(outliers_zscore)

            st.write(f"\nOutliers found for '{var}' variable using IQR method:")
            st.write(outliers_iqr)
        else:
            st.write(f"")

    data['Date'] = pd.to_datetime(data['Date'])
    data['Day'] = data['Date'].dt.day
    data['Month'] = data['Date'].dt.month
    data['Year'] = data['Date'].dt.year
    data.drop(columns=['Date'], inplace = True)

def data_visualization(data):
    # Create separate histograms for 'Age' and 'Month' based on total sales
    # Create a button to toggle between Age and Month distributions
    st.subheader('Data Visualization')
    col1, col2 = st.columns(2)

    # First column: Age distribution
    with col1:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(data['Age'], bins=10, color='skyblue', edgecolor='black')
        ax.set_title('Distribution of Age')
        ax.set_xlabel('Age')
        ax.set_ylabel('Frequency')
        st.pyplot(fig)

    # Second column: Month distribution
    with col2:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(data['Month'], bins=12, color='salmon', edgecolor='black')
        ax.set_title('Distribution of Month')
        ax.set_xlabel('Month')
        ax.set_ylabel('Frequency')
        st.pyplot(fig)

    monthly_data = data.groupby('Month').agg({'Quantity': 'sum', 'Total Amount': 'sum'}).reset_index()

    # Create a line plot for 'Month' and 'Quantity' sold
    fig, ax1 = plt.subplots(figsize=(20, 6))

    # Plot 'Quantity' sold
    ax1.plot(monthly_data['Month'], monthly_data['Quantity'], marker='o', color='skyblue', label='Quantity Sold')
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Quantity Sold', color='skyblue')
    ax1.tick_params(axis='y', labelcolor='skyblue')

    # Create a second y-axis for 'Total Amount'
    ax2 = ax1.twinx()
    ax2.plot(monthly_data['Month'], monthly_data['Total Amount'], marker='x', color='salmon', label='Total Amount')
    ax2.set_ylabel('Total Amount', color='salmon')
    ax2.tick_params(axis='y', labelcolor='salmon')

    # Set plot title and grid
    plt.title('Monthly Quantity Sold and Total Amount')
    plt.grid(True)

    # Show legends
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Show plot in Streamlit
    st.pyplot(fig)

    col1, col2 = st.columns(2)

    with col1:
        category_sales = data.groupby('Product Category')['Quantity'].sum().reset_index()

        # Create a pie chart
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.pie(category_sales['Quantity'], labels=category_sales['Product Category'], autopct='%1.1f%%', startangle=140)
        ax.set_title('Distribution of Sales Across Product Categories')
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        # Show plot in Streamlit
        st.pyplot(fig)
    
    with col2:
        # Pivot the data to create a matrix where rows are months, columns are product categories, and values are quantities sold
        heatmap_data = data.pivot_table(index='Month', columns='Product Category', values='Quantity', aggfunc='sum')

        # Create a heatmap with red color
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(heatmap_data, cmap='Reds', annot=True, fmt='d', ax=ax)
        ax.set_title('Sales Quantity by Month and Product Category')
        ax.set_xlabel('Product Category')
        ax.set_ylabel('Month')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)

        # Show plot in Streamlit
        st.pyplot(fig)
        
    fig1, ax1 = plt.subplots(figsize=(15, 6))
    sns.countplot(x='Product Category', hue='Gender', data=data, palette='colorblind', ax=ax1)
    ax1.set_title('Product Category by Gender')
    ax1.set_xlabel('Product Category')
    ax1.set_ylabel('Count')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)

    # Show the first plot in Streamlit
    st.pyplot(fig1)

    # Plot to understand which age category buys which product
    fig2, ax2 = plt.subplots(figsize=(20, 6))
    sns.countplot(x='Product Category', hue='Age', data=data, ax=ax2)
    ax2.set_title('Product Category by Age')
    ax2.set_xlabel('Product Category')
    ax2.set_ylabel('Count')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
    ax2.legend(title='Age')

    # Show the second plot in Streamlit
    st.pyplot(fig2)

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
    st.title('Retail Stock Analysis')
    #if st.button('Press to use Example Dataset'):
    df = pd.read_csv('data_files/retail.csv')
    st.subheader('Data Processing')
    st.write(df.head(5))
    data_process(df)
    data_visualization(df)
    build_model(df)
    conclusion()


if __name__ == "__main__":
    app()
