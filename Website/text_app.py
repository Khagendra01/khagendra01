import streamlit as st
import json
import pandas as pd
from tensorflow import keras

st.title("Machine Learning Algorithm Tester")
st.write("Authored by Kgen, Shadip")

model = keras.models.load_model("text_model.h5")

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

    char_arr = [[] for _ in range(int(max(df['u'])) + 1)]

    if '^' in df.columns:
        for char, cellNumber, val in zip(df['^'], df['u'].fillna(method='ffill'), df['v']):
            cellNumber = int(cellNumber)       
            if isinstance(char, list):
                if len(char) >= 2:
                    char_arr[cellNumber].append(char[2])
            if not isinstance(val, float):
                if val != "":
                    char_arr[cellNumber].append(char)
    else:
        for char, cellNumber in zip(df['v'], df['u'].fillna(method='ffill')):
            cellNumber = int(cellNumber)
            if isinstance(char, str):
                char_arr[cellNumber].append(char)

    #char_arr[cellNumber].insert(char[0],char[2])
    flattened_arr = [item for sublist in char_arr for item in sublist]

    original_length = len(flattened_arr)
    desired_length = 75711

    padding_length = desired_length - original_length

    data_set += [0] * padding_length

    X = []
    X.append(data_set)

    y_pred_numb_binary_flat = (model.predict(X).flatten() > 0.5).astype(int)
    st.write(y_pred_numb_binary_flat)
    

else:
    st.write("Please upload a file.")
