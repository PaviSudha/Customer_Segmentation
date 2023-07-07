from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load the model
classifier = pickle.load(open('classifier.pkl', 'rb'))

# Customer segmentation function
def segment_customers(input_data):
    input_df = pd.DataFrame([input_data], columns=["Age", "Education", "Parental_Status", "kids",
                                                  "Income", "Average_Spent", "Customer_Loyalty",
                                                  "Discount_Purchases", "Total_Promo"])
    prediction = classifier.predict(input_df)
    return prediction[0]

@app.route('/', methods=['GET', 'POST'])
def index():
    result = ""
    if request.method == 'POST':
        age = int(request.form['age'])
        education = int(request.form['education'])
        parental_status = int(request.form['parental_status'])
        kids = int(request.form['kids'])
        income = int(request.form['income'])
        average_spent = int(request.form['average_spent'])
        customer_loyalty = int(request.form['customer_loyalty'])
        discount_purchases = int(request.form['discount_purchases'])
        total_promo = int(request.form['total_promo'])

        input_data = [age, education, parental_status, kids, income, average_spent,
                      customer_loyalty, discount_purchases, total_promo]

        prediction = segment_customers(input_data)
        result = f"The customer belongs to cluster {prediction}"

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
