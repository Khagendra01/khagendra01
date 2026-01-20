import streamlit as st
import json
import pandas as pd
from tensorflow import keras

st.title("Machine Learning Algorithm Tester")
st.write("Authored by Kgen, Shadip")

model = keras.models.load_model("numb_model.h5")

uploaded_file = st.file_uploader("Upload your assignment file", type=['submit'])

if uploaded_file is not None:
    st.write("File uploaded successfully!")

    json_data = json.load(uploaded_file)
    log_field = json_data['submission']['logs'][0]['log']
    logs = log_field.strip().split('\n')

    log_list = []

    for log in logs:
        data = json.loads(log)
        log_list.append(data)

    df = pd.DataFrame(log_list)

    data_set = []
    cnt = 0
    # Processing the DataFrame skipping the first entry
    data_set = []
    cnt = 0
    # Processing the DataFrame skipping the first entry
    # Refactor: Replace deprecated fillna(method='ffill') with ffill()
    df_filled = df.copy()
    df_filled['_cs'] = df_filled['_cs'].ffill()
    df_filled['_c'] = df_filled['_c'].ffill()

    # Iterate starting from index 1
    for i in range(1, len(df_filled)):
        total_chars = df_filled['_cs'].iloc[i]
        cursor_position = df_filled['_c'].iloc[i]
        
        if pd.notna(total_chars) and pd.notna(cursor_position):
             data_set.append(cnt)
             data_set.append(int(total_chars))
             data_set.append(int(cursor_position))
             cnt += 1

    X = [data_set]

    y_pred_numb_binary_flat = (model.predict(X).flatten() > 0.5).astype(int)
    st.write(y_pred_numb_binary_flat)
    

else:
    st.write("Please upload a file.")
