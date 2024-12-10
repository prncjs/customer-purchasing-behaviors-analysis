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
    
    This dataset is designed to help analyze customer segmentation, loyalty trends, and purchasing behavior. By examining the relationships between the attributes, 
    we can better understand how different factors such as age, income, and loyalty impact the total purchase amount. 
    Insights derived from this analysis can assist businesses in creating more targeted marketing strategies and improving customer engagement.
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
    st.write(df.drop(columns=['user_id']).describe())


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
def analysis_and_insights_section(data):
    
    tab1, tab2, tab3, tab4 = st.tabs(["Purchase Amount Analysis", "Purchase Frequency Analysis", "Key Patterns & Trends", "Anomalies & Considerations"])

    # Purchase Amount Analysis
    with tab1:
        st.subheader("Purchase Amount Analysis: Relationship with Independent Variables")

        # 'Age vs Purchase Amount' plot with correlation coefficient
        fig_age_amount, ax_age_amount = plt.subplots(figsize=(10, 6))
        sns.regplot(data=data, x='age', y='purchase_amount', scatter_kws={'alpha':0.7}, ax=ax_age_amount)
        corr_age_amount = data['age'].corr(data['purchase_amount'])
        ax_age_amount.text(0.95, 0.05, f'Corr: {corr_age_amount:.2f}', transform=ax_age_amount.transAxes, 
                        verticalalignment='bottom', horizontalalignment='right', color='blue', fontsize=10, weight='bold')
        ax_age_amount.set_title('Age vs Purchase Amount')
        st.pyplot(fig_age_amount)

        st.write("""
        - Age vs Purchase Amount
            -  The positive correlation (0.99) suggests that older customers tend to spend more. The regression line indicates a strong linear relationship, with minimal scatter around the line.
        """)

        # 'Annual Income vs Purchase Amount' plot with correlation coefficient
        fig_income_amount, ax_income_amount = plt.subplots(figsize=(10, 6))
        sns.regplot(data=data, x='annual_income', y='purchase_amount', scatter_kws={'alpha':0.7}, ax=ax_income_amount)
        corr_income_amount = data['annual_income'].corr(data['purchase_amount'])
        ax_income_amount.text(0.95, 0.05, f'Corr: {corr_income_amount:.2f}', transform=ax_income_amount.transAxes, 
                            verticalalignment='bottom', horizontalalignment='right', color='blue', fontsize=10, weight='bold')
        ax_income_amount.set_title('Annual Income vs Purchase Amount')
        st.pyplot(fig_income_amount)

        st.write("""
        - Annual Income vs Purchase Amount
            -  A high correlation (0.98) shows that customers with higher incomes generally spend more. However, the scatterplot reveals some variability, indicating other factors may also influence spending.
        """)

        # 'Loyalty Score vs Purchase Amount' plot with correlation coefficient
        fig_loyalty_amount, ax_loyalty_amount = plt.subplots(figsize=(10, 6))
        sns.regplot(data=data, x='loyalty_score', y='purchase_amount', scatter_kws={'alpha':0.7}, ax=ax_loyalty_amount)
        corr_loyalty_amount = data['loyalty_score'].corr(data['purchase_amount'])
        ax_loyalty_amount.text(0.95, 0.05, f'Corr: {corr_loyalty_amount:.2f}', transform=ax_loyalty_amount.transAxes, 
                            verticalalignment='bottom', horizontalalignment='right', color='blue', fontsize=10, weight='bold')
        ax_loyalty_amount.set_title('Loyalty Score vs Purchase Amount')
        st.pyplot(fig_loyalty_amount)

        st.write("""
        - Loyalty Score vs Purchase Amount
            -  The strongest correlation (0.99) highlights that loyalty is a key driver of spending. The regression line is steep, showing that even small increases in loyalty score significantly impact purchase amounts.
        """)

        # 'Purchase Frequency vs Purchase Amount' plot with correlation coefficient
        fig_freq_amount, ax_freq_amount = plt.subplots(figsize=(10, 6))
        sns.regplot(data=data, x='purchase_frequency', y='purchase_amount', scatter_kws={'alpha':0.7}, ax=ax_freq_amount)
        corr_freq_amount = data['purchase_frequency'].corr(data['purchase_amount'])
        ax_freq_amount.text(0.95, 0.05, f'Corr: {corr_freq_amount:.2f}', transform=ax_freq_amount.transAxes, 
                        verticalalignment='bottom', horizontalalignment='right', color='blue', fontsize=10, weight='bold')
        ax_freq_amount.set_title('Purchase Frequency vs Purchase Amount')
        st.pyplot(fig_freq_amount)

        st.write("""
        - Purchase Frequency vs Purchase Amount
            -  A strong correlation (0.99) indicates that frequent shoppers spend more. The scatterplot shows a tight clustering around the regression line, reinforcing the predictive power of frequency.
        """)


    # Purchase Frequency Analysis
    with tab2:
        st.subheader("Purchase Frequency Analysis: Relationship with Independent Variables")

        # 'Age vs Purchase Frequency' plot with correlation coefficient
        fig_age_freq, ax_age_freq = plt.subplots(figsize=(10, 6))
        sns.regplot(data=data, x='age', y='purchase_frequency', scatter_kws={'alpha':0.7}, ax=ax_age_freq)
        corr_age_freq = data['age'].corr(data['purchase_frequency'])
        ax_age_freq.text(0.95, 0.05, f'Corr: {corr_age_freq:.2f}', transform=ax_age_freq.transAxes, 
                        verticalalignment='bottom', horizontalalignment='right', color='blue', fontsize=10, weight='bold')
        ax_age_freq.set_title('Age vs Purchase Frequency')
        st.pyplot(fig_age_freq)

        st.write("""
        - Age vs Purchase Frequency
            - The correlation (0.98) suggests that older customers shop more frequently. The scatterplot shows a clear upward trend, with some outliers indicating variability in shopping habits.
        """)


        # 'Annual Income vs Purchase Frequency' plot with correlation coefficient
        fig_income_freq, ax_income_freq = plt.subplots(figsize=(10, 6))
        sns.regplot(data=data, x='annual_income', y='purchase_frequency', scatter_kws={'alpha':0.7}, ax=ax_income_freq)
        corr_income_freq = data['annual_income'].corr(data['purchase_frequency'])
        ax_income_freq.text(0.95, 0.05, f'Corr: {corr_income_freq:.2f}', transform=ax_income_freq.transAxes, 
                            verticalalignment='bottom', horizontalalignment='right', color='blue', fontsize=10, weight='bold')
        ax_income_freq.set_title('Annual Income vs Purchase Frequency')
        st.pyplot(fig_income_freq)

        st.write("""
        - Annual Income vs Purchase Frequency
            - A correlation of 0.98) suggests that higher-income customers shop more often. The regression line is well-fitted, but some scatter indicates other influencing factors.
        """)


        # 'Loyalty Score vs Purchase Frequency' plot with correlation coefficient
        fig_loyalty_freq, ax_loyalty_freq = plt.subplots(figsize=(10, 6))
        sns.regplot(data=data, x='loyalty_score', y='purchase_frequency', scatter_kws={'alpha':0.7}, ax=ax_loyalty_freq)
        corr_loyalty_freq = data['loyalty_score'].corr(data['purchase_frequency'])
        ax_loyalty_freq.text(0.95, 0.05, f'Corr: {corr_loyalty_freq:.2f}', transform=ax_loyalty_freq.transAxes, 
                            verticalalignment='bottom', horizontalalignment='right', color='blue', fontsize=10, weight='bold')
        ax_loyalty_freq.set_title('Loyalty Score vs Purchase Frequency')
        st.pyplot(fig_loyalty_freq)

        st.write("""
        - Loyalty Score vs Purchase Frequency
            - A strong correlation (0.99) shows that loyal customers shop more frequently. The scatterplot is tightly clustered, indicating a consistent relationship.
       """)


        # 'Purchase Amount vs Purchase Frequency' plot with correlation coefficient
        fig_amount_freq, ax_amount_freq = plt.subplots(figsize=(10, 6))
        sns.regplot(data=data, x='purchase_amount', y='purchase_frequency', scatter_kws={'alpha':0.7}, ax=ax_amount_freq)
        corr_amount_freq = data['purchase_amount'].corr(data['purchase_frequency'])
        ax_amount_freq.text(0.95, 0.05, f'Corr: {corr_amount_freq:.2f}', transform=ax_amount_freq.transAxes, 
                            verticalalignment='bottom', horizontalalignment='right', color='blue', fontsize=10, weight='bold')
        ax_amount_freq.set_title('Purchase Amount vs Purchase Frequency')
        st.pyplot(fig_amount_freq)


        st.write("""
        - Purchase Amount vs Purchase Frequency
            - The highest correlation (0.99) confirms that higher spending is associated with more frequent shopping. The scatterplot shows a near-perfect linear relationship.
        """)

    with tab3:
        st.subheader('Key Patterns and Trends')

        st.write("""
        - Strong Relationships Across Variables:
            - Both purchase amount and purchase frequency show strong positive correlations with age, income, loyalty score, and each other. This indicates that these variables are interrelated and collectively influence customer behavior.
        - Loyalty as a Key Driver:
            - Loyalty score consistently shows the strongest correlations with both purchase amount and purchase frequency. This highlights the critical role of customer loyalty in driving revenue and engagement.
        - Age and Income as Predictors:
            - Age and income are strong predictors of both purchase amount and frequency, reflecting the importance of demographic and financial factors in shaping customer behavior.
        - Purchase Frequency and Amount Interdependence:
            - The near-perfect correlation between purchase amount and frequency suggests that customers who shop more often also tend to spend more, reinforcing the importance of encouraging repeat purchases.
        """)

    with tab4:
        st.subheader('Anomalies and Considerations')

        st.write("""
        - Outliers:
            - While the relationships are strong, some scatterplots show outliers, indicating variability in customer behavior. These outliers could represent unique customer segments or external factors influencing spending and shopping habits.
        - Other Influencing Factors:
            - Despite the strong correlations, some variability suggests that other factors (e.g., product preferences, marketing effectiveness, or external economic conditions) may also play a role.
        """)

        

        

# Conclusions and Recommendations Section
def conclusions_and_recommendations_section():
    st.subheader('Conclusions and Recommendations')

    # Key Insights and Recommendations
    insights_recommendations = {
        "Age vs Purchase Amount": {
            "Insight": "Older customers tend to spend more, as evidenced by the strong positive correlation.",
            "Recommendations": [
                "**Age-Based Strategies**",
                "- Develop premium products/services for older customers.",
                "- Create targeted marketing campaigns for different age groups."
            ]
        },
        "Annual Income vs Purchase Amount": {
            "Insight": "Higher income is associated with greater spending, although some variability exists.",
            "Recommendations": [
                "**Income-Based Strategies**",
                "- Offer exclusive, high-value products for higher-income customers.",
                "- Implement tiered pricing strategies to cater to different income levels."
            ]
        },
        "Loyalty Score vs Purchase Amount": {
            "Insight": "Loyal customers spend significantly more, showing the highest correlation.",
            "Recommendations": [
                "**Loyalty Program Enhancements**",
                "- Strengthen rewards for high-spending customers.",
                "- Create personalized offers to encourage larger purchases."
            ]
        },
        "Purchase Frequency vs Purchase Amount": {
            "Insight": "Frequent shoppers spend more, with a tightly clustered linear relationship.",
            "Recommendations": [
                "**Encourage Repeat Purchases**",
                "- Use promotions and discounts to incentivize frequent shoppers.",
                "- Develop subscription-based models for consistent revenue."
            ]
        },
        "Age vs Purchase Frequency": {
            "Insight": "Older customers shop more frequently, showing an upward trend.",
            "Recommendations": [
                "**Age-Based Engagement**",
                "- Design campaigns that appeal to older customers' preferences.",
                "- Offer incentives for younger customers to shop more frequently."
            ]
        },
        "Annual Income vs Purchase Frequency": {
            "Insight": "Higher-income customers tend to shop more often, with some variability.",
            "Recommendations": [
                "**Income-Based Engagement**",
                "- Provide exclusive benefits for high-income customers to increase visit frequency.",
                "- Use targeted ads to attract middle-income customers."
            ]
        },
        "Loyalty Score vs Purchase Frequency": {
            "Insight": "Loyal customers shop more frequently, with a strong positive correlation.",
            "Recommendations": [
                "**Loyalty Program Optimization**",
                "- Reward frequent visits with points or discounts.",
                "- Create tiered loyalty levels to encourage more frequent shopping."
            ]
        },
        "Purchase Amount vs Purchase Frequency": {
            "Insight": "Higher spending is associated with more frequent shopping, showing a near-perfect relationship.",
            "Recommendations": [
                "**Cross-Selling Opportunities*",
                "- Promote complementary products to increase purchase frequency.",
                "- Use data-driven recommendations to personalize shopping experiences."
            ]
        }
    }

    # Dropdown for insights exploration
    selected_relationship = st.selectbox(
        "Select a Relationship to Explore Insights",
        list(insights_recommendations.keys())
    )

    # Display the insight and recommendations
    if selected_relationship:
        st.subheader(f"Insights for {selected_relationship}")
        st.write(f"**Insight:** {insights_recommendations[selected_relationship]['Insight']}")
        st.write("**Recommendations:**")
        for rec in insights_recommendations[selected_relationship]["Recommendations"]:
            st.markdown(rec)

    

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
