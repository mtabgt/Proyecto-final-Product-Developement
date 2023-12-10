import streamlit as st
import requests
import json
import pandas as pd


api_url = "http://127.0.0.1:5000" # URL para acceder a API

# Función para enviar datos de web app a modelo individual
def predict_single_model(model_route, data): # Se envía la ruta del modelo al que se desea acceder, y la data recabada
    response = requests.post(f"{api_url}/{model_route}", json=data) # Se recibe el resultado del modelo en formato JSON
    return response.json()


# Función para enviar datos de web app a modelo múltile
def predict_multiple_model(model_route, data): # Se envía la ruta del modelo al que se desea acceder, y la data recabada
    response = requests.post(f"{api_url}/{model_route}", json={"data": data})# Se recibe el resultado del modelo en formato JSON
    return response.json()


# Función para recabar información en web app y aplicación de modelo múltiple (transformación de datos y creación de dataset para enviar a API)
def collect_data(num_datasets, model_number):
    data_multiple_model = {"data": []}

    for i in range(num_datasets): # Se crea un for para tener una clave de cada dataset y evitar duplicidad de registros e inconsistencias por sus llaves
        # Variables numéricas con claves únicas
        age = st.slider(f"Edad - Conjunto {i + 1}", 1, 150, 40, key=f"age_{i}")
        fnlwgt = st.slider(f"Cantidad de personas - Conjunto {i + 1}", 10000, 2000000, 203488, key=f"fnlwgt_{i}")
        educational_num = st.slider(f"Años de Educación - Conjunto {i + 1}", 0, 40, 10, key=f"educational_num_{i}")
        capital_gain = st.slider(f"Ganancias - Conjunto {i + 1}", 0, 1000000, 15024, key=f"capital_gain_{i}")
        capital_loss = st.slider(f"Pérdidas - Conjunto {i + 1}", 0, 5000, 1902, key=f"capital_loss_{i}")
        hours_per_week = st.slider(f"Horas laboradas - Conjunto {i + 1}", 0, 168, 90, key=f"hours_per_week_{i}")

        # Variables categóricas con claves únicas
        workclass = st.selectbox(f"Tipo de Trabajo - Conjunto {i + 1}", ["Private", "Self-emp-not-inc", "Local-gov", "State-gov", "Self-emp-inc", "Federal-gov", "Without-pay", "Never-worked"], key=f"workclass_{i}")
        education = st.selectbox(f"Nivel de Educación - Conjunto {i + 1}", ["Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th", "Masters", "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool"], key=f"education_{i}")
        marital_status = st.selectbox(f"Estado Civil - Conjunto {i + 1}", ["Married-civ-spouse", "Never-married", "Divorced", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"], key=f"marital_status_{i}")
        occupation = st.selectbox(f"Ocupación - Conjunto {i + 1}", ["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"], key=f"occupation_{i}")
        relationship = st.selectbox(f"Relación familiar - Conjunto {i + 1}", ["Husband", "Not-in-family", "Own-child", "Unmarried", "Wife", "Other-relative"], key=f"relationship_{i}")
        race = st.selectbox(f"Raza - Conjunto {i + 1}", ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"], key=f"race_{i}")
        gender = st.selectbox(f"Género - Conjunto {i + 1}", ["Female", "Male"], key=f"gender_{i}")
        native_country = st.selectbox(f"País de Origen - Conjunto {i + 1}", ["United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany", "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece", "South", "China", "Cuba", "Iran", "Honduras", "Philippines", "Italy", "Poland", "Jamaica", "Vietnam", "Mexico", "Portugal", "Ireland", "France", "Dominican-Republic", "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia", "Hungary", "Guatemala", "Nicaragua", "Scotland", "Thailand", "Yugoslavia", "El-Salvador", "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands"], key=f"native_country_{i}")

        # Se crea dataset con la data recabada (representa una fila)
        data_set = {
            "age": age,
            "fnlwgt": fnlwgt,
            "educational-num": educational_num,
            "capital-gain": capital_gain,
            "capital-loss": capital_loss,
            "hours-per-week": hours_per_week,
            "workclass": workclass,
            "education": education,
            "marital-status": marital_status,
            "occupation": occupation,
            "relationship": relationship,
            "race": race,
            "gender": gender,
            "native-country": native_country
        }


        
        data_set_json = json.dumps(data_set) # Se convierte el diccionario a una cadena JSON con comillas dobles
        data_set_dict = json.loads(data_set_json) # Se convierte la cadena JSON de vuelta a un diccionario
        data_multiple_model["data"].append(data_set_dict) # Se añade el diccionario modificado a la lista que se envía a API

    return data_multiple_model



def main(): # Se diseña web app
    st.title("Clasificación de ingresos salariales anuales con base en factores socioeconómicos")
    st.subheader("Mario Tabarini - 22000349")
    st.subheader("David Tejeda - 09170350")
    st.subheader("Juan Hernández - 9710120")

    model_type = st.sidebar.radio("Selecciona el tipo de modelo", ["Individual", "Multiple"]) # Sección para seleccionar tipo de modelo

    st.subheader("Ingresa tus datos para predecir:") # Sección para ingresar datos

    if model_type == "Individual":

        # Sección para seleccionar modelo individual
        model_number = st.sidebar.selectbox("Selecciona el modelo", ["First", "Second", "Third"])

        # Variables numéricas
        age = st.slider("Edad", 1, 150, 40)
        fnlwgt = st.slider("Cantidad de personas a las que representa esta encuesta (representativo)", 10000, 2000000, 203488)
        educational_num = st.slider("Años de Educación", 0, 40, 10)
        capital_gain = st.slider("Ganancias de otras inversiones adicionales al sueldo", 0, 1000000, 15024)
        capital_loss = st.slider("Pérdidas de otras inversiones adicionales al sueldo", 0, 5000, 1902)
        hours_per_week = st.slider("Horas laboradas por semana", 0, 168, 90)

        workclass = st.selectbox("Tipo de Trabajo", ["Private", "Self-emp-not-inc", "Local-gov", "State-gov", "Self-emp-inc", "Federal-gov", "Without-pay", "Never-worked"])
        education = st.selectbox("Nivel de Educación", ["Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th", "Masters", "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool"])
        marital_status = st.selectbox("Estado Civil", ["Married-civ-spouse", "Never-married", "Divorced", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"])
        occupation = st.selectbox("Ocupación", ["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"])
        relationship = st.selectbox("Relación familiar de la persona que responde a esta encuesta", ["Husband", "Not-in-family", "Own-child", "Unmarried", "Wife", "Other-relative"])
        race = st.selectbox("Raza", ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"])
        gender = st.selectbox("Género", ["Female", "Male"])
        native_country = st.selectbox("País de Origen", ["United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany", "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece", "South", "China", "Cuba", "Iran", "Honduras", "Philippines", "Italy", "Poland", "Jamaica", "Vietnam", "Mexico", "Portugal", "Ireland", "France", "Dominican-Republic", "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia", "Hungary", "Guatemala", "Nicaragua", "Scotland", "Thailand", "Yugoslavia", "El-Salvador", "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands"])

        data_single_model = {
            "age": age,
            "fnlwgt": fnlwgt,
            "educational-num": educational_num,
            "capital-gain": capital_gain,
            "capital-loss": capital_loss,
            "hours-per-week": hours_per_week,
            "workclass": workclass,
            "education": education,
            "marital-status": marital_status,
            "occupation": occupation,
            "relationship": relationship,
            "race": race,
            "gender": gender,
            "native-country": native_country
        }

        # Código para realizar la predicción individual
        if st.button("Realizar predicción de ingresos"):
            result_single_model = predict_single_model(f"predictOne{model_number}Model", data_single_model)
            df_resultados = pd.DataFrame({'Predicciones de ingresos (en USD)': result_single_model}) # Se crea df para imprimirlo en una tabla
            st.table(df_resultados) # Se imprime resultado en tabla


    elif model_type == "Multiple": # Código si se seleccionó la opción de predicción múltiple
        num_datasets = st.sidebar.number_input("Número de Conjuntos de Datos", min_value=1, value=1) # Input para selección de cantidad de filas que se desea predecir
        model_number = st.sidebar.selectbox("Selecciona el modelo", ["First", "Second", "Third"]) # Selectbox para primer, segundo o tercer modelo a aplicar

        data_multiple_model = collect_data(num_datasets, model_number) # Se crea variable con la elección del usuario sobre cantidad de filas y modelo a emplear


        if st.button("Realizar Predicción Múltiple") and data_multiple_model["data"]: #Botón para desarrollar predicción múltile
            url = f'http://127.0.0.1:5000/predictMultiple{model_number}Model' # URL de modelo con variable conforme a selección del usuario
            headers = {'Content-Type': 'application/json'}
            
            response = requests.post(url, json=data_multiple_model, headers=headers) # Se envían datos hacia la API

            if response.status_code == 200:
                result_multiple_model = response.json() # Se obienen los resultados de la API si existen
                df_resultados = pd.DataFrame(result_multiple_model) # Se crea un df para imprimir resultados en tabla
                st.table(df_resultados) # Se imprimen resultados en tabla
            else:
                st.error(f"Error en la solicitud: {response.status_code}")



if __name__ == "__main__":
    main()