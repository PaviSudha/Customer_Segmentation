import pickle
import streamlit as st
import pandas as pd

# Load the model
classifier = pickle.load(open('classifier.pkl', 'rb'))

# Page configuration
st.set_page_config(page_title='Customer Segmentation Web App', layout='centered')
st.title('Customer Segmentation Web App')

# Customer segmentation function
def segment_customers(input_data):
    input_df = pd.DataFrame([input_data], columns=["Age", "Education", "Parental_Status", "kids",
                                                  "Income", "Average_Spent", "Customer_Loyalty",
                                                  "Discount_Purchases", "Total_Promo"])
    prediction = classifier.predict(input_df)
    return prediction[0]

def main():
    age = st.slider("Select Age", 18, 85)
    education_options = {0: "Undergraduate", 1: "Graduate", 2: "Post Graduate"}
    education = st.selectbox("Select Education", options=list(education_options.keys()), format_func=lambda x: education_options[x])
    parental_options = {0: "No Children", 1: "Having Children"}
    parental_status = st.selectbox("Select Parental Status", options=list(parental_options.keys()), format_func=lambda x: parental_options[x])
    kids = st.number_input("Enter the Number of Kids in Household", min_value=0, max_value=3, step=1)
    income = st.number_input("Enter the Household Income")
    average_spent = st.number_input("Enter the Average Spending of the Customer")
    customer_loyalty = st.number_input("Enter the Number of Days Customer Present in the Company")
    discount_purchases = st.number_input("Enter the Discount Purchases of the Customer")
    total_promo = st.number_input("Enter the Number of Promotions Accepted by the Customer", min_value=0, max_value=6, step=1)

    result = ""

    # When 'Segment Customer' button is clicked, make the prediction and display the result
    if st.button("Segment Customer"):
        input_data = [age, education,parental_status, kids, income, average_spent,
                      customer_loyalty,
                      discount_purchases, total_promo]
        prediction = segment_customers(input_data)
        result = f"The customer belongs to cluster {prediction}"

    st.success(result)

if __name__ == '__main__':
    main()
