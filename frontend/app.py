import os
import pandas as pd
import requests
import streamlit as st

# Backend URL — inside Docker network, containers see each other by service name
BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:7860")

# Same constants used at training time
CURRENT_YEAR = 2026
PERISHABLES = ["Dairy", "Meat", "Fruits and Vegetables", "Breads",
               "Breakfast", "Frozen Foods", "Seafood"]


# Page setup
st.set_page_config(page_title="SuperKart Sales Prediction",
                   page_icon="🛒",
                   layout="centered")

st.title("🛒 SuperKart Sales Prediction")
st.write("Predicts quarterly sales revenue for a product at a SuperKart store.")


# ---------- Single prediction ----------
st.subheader("Single Prediction")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Product**")
    product_id = st.text_input("Product ID (e.g. FD1234, DR5678, NC9012)", value="FD1234")
    product_weight = st.slider("Product weight", 4.0, 22.0, 12.5, 0.1)
    product_sugar = st.selectbox("Sugar content", ["Regular", "Low Sugar", "No Sugar"])
    product_area = st.slider("Allocated store area (fraction)", 0.001, 0.30, 0.05, 0.001)
    product_type = st.selectbox("Product type", [
        "Dairy", "Meat", "Fruits and Vegetables", "Breads", "Breakfast",
        "Frozen Foods", "Seafood", "Baking Goods", "Canned", "Snack Foods",
        "Health and Hygiene", "Household", "Hard Drinks", "Soft Drinks",
        "Starchy Foods", "Others"
    ])
    product_mrp = st.slider("MRP", 30.0, 270.0, 130.0, 1.0)

with col2:
    st.markdown("**Store**")
    store_year = st.number_input("Establishment year", 1985, CURRENT_YEAR, 2010, 1)
    store_size = st.selectbox("Store size", ["Small", "Medium", "High"])
    store_city = st.selectbox("City tier", ["Tier 1", "Tier 2", "Tier 3"])
    store_type = st.selectbox("Store type", [
        "Departmental Store", "Food Mart",
        "Supermarket Type1", "Supermarket Type2"
    ])


# When the user clicks Predict
if st.button("Predict quarterly sales", type="primary"):

    # Do the feature engineering the model expects
    product_id_char = product_id[:2].upper()
    store_age = CURRENT_YEAR - int(store_year)

    if product_type in PERISHABLES:
        product_type_category = "Perishables"
    else:
        product_type_category = "Non Perishables"

    # Build the payload
    payload = {
        "Product_Weight": product_weight,
        "Product_Sugar_Content": product_sugar,
        "Product_Allocated_Area": product_area,
        "Product_MRP": product_mrp,
        "Store_Size": store_size,
        "Store_Location_City_Type": store_city,
        "Store_Type": store_type,
        "Product_Id_char": product_id_char,
        "Store_Age_Years": store_age,
        "Product_Type_Category": product_type_category
    }

    # Send the request
    try:
        response = requests.post(BACKEND_URL + "/v1/sales", json=payload, timeout=10)

        if response.status_code == 200:
            result = response.json()
            predicted_sales = result["Predicted_Sales"]
            st.success(f"Predicted quarterly sales: ₹{predicted_sales:,.2f}")
        else:
            st.error(f"API returned {response.status_code}: {response.text}")

    except requests.RequestException as e:
        st.error(f"Could not reach backend: {e}")


st.divider()


# ---------- Batch prediction ----------
st.subheader("Batch Prediction")
st.caption("Upload a CSV with the ten engineered feature columns.")

uploaded_file = st.file_uploader("CSV file", type=["csv"])

if uploaded_file is not None:
    if st.button("Predict for batch", type="primary"):
        try:
            # Send the file to the backend
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
            response = requests.post(BACKEND_URL + "/v1/salesbatch", files=files, timeout=30)

            if response.status_code == 200:
                results = pd.DataFrame(response.json())
                st.write(f"Received predictions for {len(results)} records.")
                st.dataframe(results)

                # Let the user download the results
                csv_data = results.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download predictions as CSV",
                    data=csv_data,
                    file_name="superkart_predictions.csv",
                    mime="text/csv"
                )
            else:
                st.error(f"API returned {response.status_code}: {response.text}")

        except requests.RequestException as e:
            st.error(f"Could not reach backend: {e}")
