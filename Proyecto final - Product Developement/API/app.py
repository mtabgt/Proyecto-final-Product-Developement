from flask import Flask, request, jsonify
import pandas as pd
from pycaret.regression import load_model
import json
from pycaret.regression import predict_model
from datetime import datetime

# Se cargan los tres modelos generados en el ipynb
model_1=load_model('../models/model_exp1') 
model_2=load_model('../models/model_exp2')
model_3=load_model('../models/model_exp3')
app=Flask(__name__)


# Creando funciones para aplicar predicciones de los datos en formato JSON

@app.route('/predictOneFirstModel', methods=['POST']) # Se aplica POST para enviar datos obtenidos en web app
def predictOneFirstModel(): # Función para aplicar predicción individual conforme a primer modelo
    data=request.json # Se guarda data obtenida en variable "data"
    data_to_predict=pd.json_normalize(data) # Se normalza la data en formato JSON
    try: # Manejo de excepciones
        prediccion=predict_model(model_1, data=data_to_predict) # Se aplica modelo 1 del notebook con la data obtenida
        valor_predicho=list(prediccion['prediction_label'])[0] # Se recibe el resultado en forma de lista
        current_date=datetime.now() # Se obtiene la fecha de hoy para log
        with open('model_logs.log', 'a') as archivo_modificado: # Se crea un nuevo registro en el log (se actualiza el archivo)
            strLog=f'Model: lightgbm (model 1) - Predicted_Value:{valor_predicho} - Date:{current_date.strftime("%Y-%m-%d %H:%M:%S")}' # Se añaden los datos relevantes al log
            archivo_modificado.write(strLog + '\n') # Se guarda el nuevo registro en log

        print(valor_predicho)
        return jsonify({'Predicción conforme a primer modelo individual':valor_predicho}) # Se retorna el resultado en formato JSON
    except: # Si hay algún inconveniente, se maneja con la excepción
        with open('model_logs.log', 'a') as archivo_modificado:
            strLog=f'Error:{current_date.strftime("%Y-%m-%d%H:%M:%S")}'
            archivo_modificado.write(strLog + '\n')
        return jsonify({'mensaje': "Se generó un error en la predicción, revise los datos ingresados."})

# Misma funcionalidad que código anterior, aplicada para segundo modelo individual
@app.route('/predictOneSecondModel', methods=['POST'])
def predictOneSecondModel():
    data=request.json
    data_to_predict=pd.json_normalize(data)
    try:
        prediccion=predict_model(model_2, data=data_to_predict)
        valor_predicho=list(prediccion['prediction_label'])[0]
        current_date=datetime.now()
        with open('model_logs.log', 'a') as archivo_modificado:
            strLog=f'Model: lightgbm (model 2) - Predicted_Value:{valor_predicho} - Date:{current_date.strftime("%Y-%m-%d %H:%M:%S")}'
            archivo_modificado.write(strLog + '\n')

        print(valor_predicho)
        return jsonify({'Predicción conforme a segundo modelo individual':valor_predicho})
    except:
        with open('model_logs.log', 'a') as archivo_modificado:
            strLog=f'Error:{current_date.strftime("%Y-%m-%d%H:%M:%S")}'
            archivo_modificado.write(strLog + '\n')
        return jsonify({'mensaje': "Se generó un error en la predicción, revise los datos ingresados."})


# Misma funcionalidad que código anterior, aplicada para tercer modelo individual
@app.route('/predictOneThirdModel', methods=['POST'])
def predictOneThirdModel():
    data=request.json
    data_to_predict=pd.json_normalize(data)
    try:
        prediccion=predict_model(model_3, data=data_to_predict)
        valor_predicho=list(prediccion['prediction_label'])[0]
        current_date=datetime.now()
        with open('model_logs.log', 'a') as archivo_modificado:
            strLog=f'Model: lightgbm (model 3) - Predicted_Value:{valor_predicho} - Date:{current_date.strftime("%Y-%m-%d %H:%M:%S")}'
            archivo_modificado.write(strLog + '\n')

        print(valor_predicho)
        return jsonify({'Predicción conforme a tercer modelo individual':valor_predicho})
    except:
        with open('model_logs.log', 'a') as archivo_modificado:
            strLog=f'Error:{current_date.strftime("%Y-%m-%d%H:%M:%S")}'
            archivo_modificado.write(strLog + '\n')
        return jsonify({'mensaje': "Se generó un error en la predicción, revise los datos ingresados."})


# Código para predicciones múltiples


@app.route('/predictMultipleFirstModel', methods=['POST']) # Ruta de modelo y POST para publicación de set de datos
def predictMultipleFirstModel(): # Función para primer modelo con múltiples filas
    try:
        data = request.json.get('data', []) # Se obtiene la data enviada por la web app
        print(data)
        predictions = [] # Se crea df para almacenar arreglos

        for data_point in data: # Se recorre conjunto de datos JSON
            try:
                data_to_predict = pd.json_normalize(data_point) # Se normalizan datos a formato de python
                
                prediccion = predict_model(model_1, data=data_to_predict) # Se aplica modelo 1 para predecir datos
                print(prediccion)
                valor_predicho = list(prediccion['prediction_label'])[0] # Se guarda resultado en lista
                current_date = datetime.now()
                
                predictions.append(valor_predicho) # Se añade a arreglo de predicciones el resultado para la fila evaluada
                
                #Manejando excepciones
            except Exception as e:
                print(f"Error durante el procesamiento: {e}")

        return jsonify({'Predicciones de ingresos (en USD)': predictions})

    except Exception as e:
        current_date = datetime.now()
        with open('model_logs.log', 'a') as archivo_modificado: # Se añade al log la excepción, hora, fecha
            strLog = f'Error:{current_date.strftime("%Y-%m-%d %H:%M:%S")}'
            archivo_modificado.write(strLog + '\n')
        return jsonify({'mensaje': f"Se generó un error en la predicción, revise los datos ingresados. Detalles: {str(e)}"})


# Misma funcionalidad que código anterior, aplicada a segundo modelo múltiple
@app.route('/predictMultipleSecondModel', methods=['POST'])
def predictMultipleSecondModel():
    try:
        data = request.json.get('data', [])
        print(data)
        predictions = []

        for data_point in data:
            try:
                data_to_predict = pd.json_normalize(data_point)
                
                prediccion = predict_model(model_2, data=data_to_predict)
                print(prediccion)
                valor_predicho = list(prediccion['prediction_label'])[0]
                current_date = datetime.now()
                
                predictions.append(valor_predicho)
                
            except Exception as e:
                print(f"Error durante el procesamiento: {e}")

        return jsonify({'Predicciones de ingresos (en USD)': predictions})

    except Exception as e:
        current_date = datetime.now()
        with open('model_logs.log', 'a') as archivo_modificado:
            strLog = f'Error:{current_date.strftime("%Y-%m-%d %H:%M:%S")}'
            archivo_modificado.write(strLog + '\n')
        return jsonify({'mensaje': f"Se generó un error en la predicción, revise los datos ingresados. Detalles: {str(e)}"})


# Misma funcionalidad que código anterior, aplicada a tercer modelo múltiple
@app.route('/predictMultipleThirdModel', methods=['POST'])
def predictMultipleThirdModel():
    try:
        data = request.json.get('data', [])
        print(data)
        predictions = []

        for data_point in data:
            try:
                data_to_predict = pd.json_normalize(data_point)
                
                prediccion = predict_model(model_3, data=data_to_predict)
                print(prediccion)
                valor_predicho = list(prediccion['prediction_label'])[0]
                current_date = datetime.now()
                
                predictions.append(valor_predicho)
                
            except Exception as e:
                print(f"Error durante el procesamiento: {e}")

        return jsonify({'Predicciones de ingresos (en USD)': predictions})

    except Exception as e:
        current_date = datetime.now()
        with open('model_logs.log', 'a') as archivo_modificado:
            strLog = f'Error:{current_date.strftime("%Y-%m-%d %H:%M:%S")}'
            archivo_modificado.write(strLog + '\n')
        return jsonify({'mensaje': f"Se generó un error en la predicción, revise los datos ingresados. Detalles: {str(e)}"})

