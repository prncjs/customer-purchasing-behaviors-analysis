import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('Customer Purchasing Behaviors.csv')
    return df

# Overview Section
def overview_section(df):
    st.subheader('Overview')

    # Introduction to the Dataset
    st.write("""
    The dataset titled **Customer Purchasing Behaviors** provides valuable insights into customer profiles and their purchasing habits.
    It includes the following key attributes:
    
    - **customer_id**: A unique identifier for each customer.
    - **age**: The age of the customer, which may influence purchasing behavior.
    - **annual_income**: The annual income (in USD) of the customer, which determines their purchasing power.
    - **purchase_amount**: The total amount spent by the customer on purchases (in USD).
    - **purchase_frequency**: The frequency of customer purchases, measured as the number of purchases per year.
    - **region**: The geographical region where the customer is located (North, South, East, West), potentially impacting their preferences and buying behavior.
    - **loyalty_score**: A score ranging from 1 to 10, representing how loyal the customer is to the brand, based on their interaction and engagement.
    
    This dataset is designed to help analyze customer segmentation, loyalty trends, and purchasing behavior. By examining the relationships between the attributes, we can better understand how different factors such as age, income, and loyalty impact the total **purchase amount**. Insights derived from this analysis can assist businesses in creating more targeted marketing strategies and improving customer engagement.
    """)

    # Research Question
    st.write("""
    **Research Question**:
    How do factors like **age**, **annual income**, and  **loyalty score** influence a customer's **purchase amount** and **purchase frequency**?

    By understanding these relationships, businesses can optimize their sales strategies and offer personalized recommendations, which could lead to increased customer retention and higher sales.
    """)

    # Selected Analysis Technique
    st.write("""
    **Selected Analysis Technique**:
    To answer the research question, we will use **Linear Regression**, a statistical method to examine the relationship between a dependent variable and one or more independent variables. In this case:
    
    - The **dependent variables** are: **Purchase Amount** and **Purchase Frequency**
    - The **independent variables** are: **Age**, **Annual Income**, **Loyalty Score**, and **Purchase Frequency** (for predicting purchase amount) / **Purchase Amount** (for predicting purchase frequency)
    
    Linear regression will help quantify the impact of each predictor on the purchase amount and purchase frequency, and evaluate how well the model explains customer purchasing behavior for both targets. 
     """)
    
    # Display dataset structure
    st.write("Dataset Structure:")
    st.write(df.head())

# Data Exploration Section
# Tab Navigation for Visualizations
def data_exploration_section(df):
    st.subheader('Data Exploration and Preparation')

    # Handle missing values
    st.write("Checking for missing values:")
    missing_values = df.isnull().sum()
    st.write(missing_values)

    if missing_values.sum() > 0:
        st.write("Filling missing values with column mean.")
        df.fillna(df.mean(), inplace=True)


    # Summary statistics
    st.write("Summary Statistics:")
    st.write(df.describe())

    st.write("Visualizations:")

    # Tab Navigation
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["Age", 
         "Annual Income", 
         "Loyalty Score",
         "Purchase Amount",  
         "Purchase Frequency",
         "Correlation Heatmap"]
    )

    with tab1: #Histogram
        st.subheader("Age")
        fig, ax = plt.subplots()
        sns.histplot(df['age'], bins=20, kde=True, color='skyblue', ax=ax)
        ax.set_title("Age Distribution")
        ax.set_xlabel("Age")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

        # Interpretation
        st.write("""
        **Interpretation**:
        - The age distribution appears to be roughly normal (bell-shaped) with a slight right skew.
        - The majority of customers are concentrated between 30-50 years old.
        - The peak of the distribution is around 35-40 years.
        - There are fewer very young customers (< 25) and older customers (> 55).
        - The distribution suggests the business appeals most to middle-aged consumers.

        """)


    with tab2:  # Annual Income
        st.subheader("Annual Income Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df['annual_income'], bins=20, kde=True, color='skyblue', ax=ax)
        ax.set_title("Annual Income Distribution")
        ax.set_xlabel("Annual Income")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)
        
        # Interpretation
        st.write("""
        **Interpretation**:
        - Shows a relatively normal distribution with some right skew.
        - The bulk of customers have annual incomes between $45,000 and $70,000.
        - The peak appears to be around $55,000-$60,000.
        - There's a longer tail towards higher incomes, indicating some high-income customers.
        - Few customers are in the very low (< $30,000) or very high (> $80,000) income brackets.
        """)
        

    with tab3:  # Loyalty Score
        st.subheader("Loyalty Score Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df['loyalty_score'], bins=20, kde=True, color='skyblue', ax=ax)
        ax.set_title("Loyalty Score Distribution")
        ax.set_xlabel("Loyalty Score")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

        st.write("""
        **Interpretation**:
        - Shows a somewhat normal distribution with a slight left skew.
        - Scores range from 1-10, with most customers falling between 5-8.
        - The peak is around 7-8, indicating generally good customer loyalty.
        - Fewer customers have very low (1-3) or very high (9-10) loyalty scores.
        - The distribution suggests successful customer retention with room for improvement in converting lower-scoring customers.
        """)

    with tab4:  # Purchase Amount
        st.subheader("Purchase Amount Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df['purchase_amount'], bins=20, kde=True, color='skyblue', ax=ax)
        ax.set_title("Purchase Amount Distribution")
        ax.set_xlabel("Purchase Amount")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)
    
        st.write("""
        **Interpretation**:
        - Exhibits a right-skewed distribution.
        - Most purchase amounts cluster between $200-$600.
        - The peak is around $300-$400.
        - There's a long tail extending towards higher purchase amounts.
        - This pattern suggests a core range of typical purchases with some customers making significantly larger purchases.
        """)

    with tab5:  # Purchase Frequency
        st.subheader("Purchase Frequency Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df['purchase_frequency'], bins=20, kde=True, color='skyblue', ax=ax)
        ax.set_title("Purchase Frequency Distribution")
        ax.set_xlabel("Purchase Frequency")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

        st.write("""
        **Interpretation**:
        - The distribution has a slight right skew, with a longer tail extending beyond 25 purchases.
        - The first peak occurs around 15 purchases, and the second peak is around 22 purchases.
        - The majority of customers have purchase frequencies between 10 and 25 purchases.
        - There is a noticeable dip in frequency between 17 and 20 purchases, indicating fewer customers in this range.
        - Very few customers have purchase frequencies below 12 purchases or above 27 purchases.
        """)

    with tab6:
        st.subheader("Correlation Heatmap")
        
        df_cleaned = df.drop(columns=['user_id', 'region'])
        
        # Create the correlation matrix
        corr_matrix = df_cleaned.corr()

        # Plot the heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
        st.pyplot(plt)

        st.write("""
        **Interpretation**:        
        - Age is highly correlated with all other variables, particularly with purchase_amount (0.99) and purchase_frequency (0.98).
        - Annual_income has a strong correlation with purchase_amount (0.98), loyalty_score (0.98), and purchase_frequency (0.98), indicating that higher income is associated with higher spending and loyalty.
        - Purchase_amount and purchase_frequency are almost perfectly correlated (0.99), suggesting that customers who spend more also shop more frequently.
        - Loyalty_score is strongly correlated with both purchase_amount (0.99) and purchase_frequency (0.99), implying that loyal customers tend to spend more and shop more often.
        - The diagonal values are 1.00, representing the perfect correlation of each variable with itself.
        - Overall, the heatmap indicates a tightly interconnected dataset, where all variables are strongly related to one another.

        """)

#Analysis and Insights

def analysis_and_insights_section(df):
    st.subheader("Analysis and Insights")

    # Predicting Purchase Amount
    X = df[['age', 'annual_income', 'loyalty_score', 'purchase_frequency']]  # Input features
    y_amount = df['purchase_amount']  # Target for purchase amount

    # Split and train the model for purchase amount
    X_train, X_test, y_train, y_test = train_test_split(X, y_amount, test_size=0.2, random_state=42)
    model_amount = LinearRegression()
    model_amount.fit(X_train, y_train)

    # Predicting Purchase Frequency
    y_freq = df['purchase_frequency']  # Target for purchase frequency

    # Split and train the model for purchase frequency
    X_train_freq, X_test_freq, y_train_freq, y_test_freq = train_test_split(X, y_freq, test_size=0.2, random_state=42)
    model_freq = LinearRegression()
    model_freq.fit(X_train_freq, y_train_freq)

    # Sidebar for selecting model type
    st.sidebar.subheader("Choose Regression Model")
    model_type = st.sidebar.selectbox(
        "Regression Type", 
        ["Linear Regression", "Ridge Regression", "Lasso Regression", "Elastic Net Regression"]
    )

    # Slider for regularization parameters
    alpha = st.sidebar.slider("Regularization Strength (Alpha)", min_value=0.01, max_value=10.0, value=1.0, step=0.1)

    # Model selection
    if model_type == "Linear Regression":
        model = LinearRegression()
    elif model_type == "Ridge Regression":
        model = Ridge(alpha=alpha)
    elif model_type == "Lasso Regression":
        model = Lasso(alpha=alpha)
    elif model_type == "Elastic Net Regression":
        l1_ratio = st.sidebar.slider("L1 Ratio (Elastic Net)", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)

    # Create tabs for different visualizations
    tab1, tab2 = st.tabs(
        ["Purchase Amount Analysis", 
         "Purchase Frequency Analysis"]
    )

    with tab1:
        # Fit the selected model on the purchase amount data
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Display model performance for purchase amount
        st.write("### Model Performance")
        st.write(f"**Mean Squared Error (MSE):** {mean_squared_error(y_test, y_pred):.2f}")
        st.write(f"**R² Score:** {r2_score(y_test, y_pred):.2f}")

        # Show coefficients for purchase amount model
        st.write("### Model Coefficients")
        coeff_df = pd.DataFrame({"Feature": X.columns, "Coefficient": model.coef_})
        st.write(coeff_df)

        # Visualization: Predicted vs. Actual (Purchase Amount)
        st.write("### Predicted vs. Actual Values")
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, alpha=0.7)
        ax.plot([y_amount.min(), y_amount.max()], [y_amount.min(), y_amount.max()], 'k--', lw=2)
        ax.set_title(f"{model_type}: Predicted vs. Actual (Purchase Amount)")
        ax.set_xlabel("Actual Purchase Amount")
        ax.set_ylabel("Predicted Purchase Amount")
        st.pyplot(fig)

    with tab2:
        # Fit the selected model on the purchase frequency data
        model.fit(X_train_freq, y_train_freq)
        y_pred_freq = model.predict(X_test_freq)

        # Display model performance for purchase frequency
        st.write("### Model Performance")
        st.write(f"**Mean Squared Error (MSE):** {mean_squared_error(y_test_freq, y_pred_freq):.2f}")
        st.write(f"**R² Score:** {r2_score(y_test_freq, y_pred_freq):.2f}")

        # Show coefficients for purchase frequency model
        st.write("### Model Coefficients")
        coeff_df_freq = pd.DataFrame({"Feature": X.columns, "Coefficient": model.coef_})
        st.write(coeff_df_freq)

        # Visualization: Predicted vs. Actual (Purchase Frequency)
        st.write("### Predicted vs. Actual Values")
        fig_freq, ax_freq = plt.subplots()
        ax_freq.scatter(y_test_freq, y_pred_freq, alpha=0.7)
        ax_freq.plot([y_freq.min(), y_freq.max()], [y_freq.min(), y_freq.max()], 'k--', lw=2)
        ax_freq.set_title(f"{model_type}: Predicted vs. Actual (Purchase Frequency)")
        ax_freq.set_xlabel("Actual Purchase Frequency")
        ax_freq.set_ylabel("Predicted Purchase Frequency")
        st.pyplot(fig_freq)


# Conclusions and Recommendations Section
def conclusions_and_recommendations_section():
    st.subheader('Conclusions and Recommendations')
    st.write("""
    
    
    **Recommendations:**
   
    """)

# Main App with Navigation
def main():
    st.title('Customer Purchasing Behavior Analysis Using Linear Regression')

    # Load dataset
    df = load_data()

    # Sidebar layout
    with st.sidebar:
        st.title("📊 Customer Analysis")

        with st.expander("🧩 Sections", True):
           
            section = st.radio("Select", ["Overview", "Data Exploration", "Analysis and Insights", "Conclusions"])

    # Display content based on selected section
    if section == "Overview":
        overview_section(df)
    elif section == "Data Exploration":
        data_exploration_section(df)
    elif section == "Analysis and Insights":
        analysis_and_insights_section(df)
    elif section == "Conclusions":
        conclusions_and_recommendations_section()

if __name__ == "__main__":
    main()
