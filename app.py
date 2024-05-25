from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load('best_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve form data
        form_data = request.form

        # Prepare data for prediction
        data = {
            'Customer ID': form_data.get('customer_id'),
            'Customer Name': form_data.get('customer_name'),
            'Purchase Date': form_data.get('purchase_date'),
            'Purchase Year': int(form_data.get('purchase_year')),
            'Purchase Month': form_data.get('purchase_month'),
            'Product Category': form_data.get('product_category'),
            'Product Price': float(form_data.get('product_price')),
            'Quantity': int(form_data.get('quantity')),
            'Total Price': float(form_data.get('total_price')),
            'Payment Method': form_data.get('payment_method'),
            'Returns': int(form_data.get('returns')),
            'Gender': form_data.get('gender'),
            'Customer Age': int(form_data.get('customer_age')),
            'Churn': int(form_data.get('churn')),
            'Return Quantity': int(form_data.get('return_quantity')),
            'Segment Tier': form_data.get('segment_tier')
        }

        # Convert data to DataFrame
        df = pd.DataFrame([data])

        # Make prediction
        prediction = model.predict(df)
        prediction_text = int(prediction[0])

        return render_template('index.html', prediction_text=prediction_text, **data)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
