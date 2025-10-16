import streamlit as st
import pandas as pd
import pickle
import pandas as pd
import numpy as np
from sklearn.tree import  DecisionTreeClassifier
# -------------------- Load Models --------------------

with open("Bank New model.pkl", "rb") as f:
    model = pickle.load(f)


# -------------------- Manual Mappings for Categorical Columns --------------------
mappings = {
    'Job': {'management':0, 'blue-collar':1, 'technician':2, 'admin.':3, 'services':4,
            'retired':5, 'self-employed':6, 'entrepreneur':7, 'unemployed':8,
            'housemaid':9, 'student':10, 'unknown':11},
    'Marital': {'married':0, 'single':1, 'divorced':2},
    'Education': {'secondary':0, 'tertiary':1, 'primary':2, 'unknown':3},
    'Default': {'no':0, 'yes':1},
    'Housing': {'no':0, 'yes':1},
    'Loan': {'no':0, 'yes':1},
    'Month': {'jan':0,'feb':1,'mar':2,'apr':3,'may':4,'jun':5,'jul':6,'aug':7,
              'sep':8,'oct':9,'nov':10,'dec':11},
    'Poutcome': {'unknown':0, 'failure':1, 'success':2, 'other':3}
}


# -------------------- Streamlit App --------------------
st.set_page_config(page_title="Bank-Term Deposit Prediction", page_icon="🏦", layout="centered")
st.title("🏦 Bank-Term Deposit Prediction App")
st.write("Enter your details below to predict if a client will subscribe to a term deposit.")

# -------------------- Collect User Input --------------------
user_input = {}
for col, mapping in mappings.items():
    user_input[col] = st.selectbox(f"{col}", list(mapping.keys()))

for col in ['Age','Balance','Day','Duration','Campaign','Pdays','Previous']:
    user_input[col] = st.number_input(f"{col}", min_value=0, value=0)

input_df = pd.DataFrame([user_input])

# -------------------- Encode Categorical Columns --------------------
for col, mapping in mappings.items():
    input_df[col] = input_df[col].map(mapping)

correct_order = ['Age','Job','Marital','Education','Default','Balance',
                 'Housing','Loan','Day','Month','Duration','Campaign',
                 'Pdays','Previous','Poutcome']
input_df = input_df[correct_order]

# -------------------- Prediction --------------------
if st.button("🔮 Predict"):
    try:
        prediction = model.predict(input_df)[0]

        
        result = "Subscribe ✅" if prediction == 1 else "Not Subscribe ❌"
        st.success(f"Prediction: {result}")

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_df)[0][1]
            st.info(f"Probability of subscribing: {proba*100:.2f}%")

    except Exception as e:
        st.error(f"❌ Prediction Error: {e}")

# -------------------- View Input Data --------------------
with st.expander("📄 View Input Data"):

    st.dataframe(input_df)







