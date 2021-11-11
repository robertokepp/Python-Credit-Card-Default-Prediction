from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, precision_recall_curve
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
from collections import OrderedDict
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
import time

# Ignorar las advertencias
warnings.filterwarnings("ignore")

# Data del incumplimiento de la tarjeta de crédito
# ID: id de cada cliente
# LIMIT_BAL: monto del credito
# SEX: genero (1=hombre , 2=mujer)
# EDUCATION: educacion (1=postgrado , 2=universidad , 3=escuela secundaria , 4=otros , 5=desconocido , 6=desconocido)
# MARRIAGE: estado marital (1=casado , 2=soltero, 3=otros)
# AGE: edad
# (-1=pago debidamente , 1=demora en el pago por 1 meses , 2=demora en el pago por 2 meses , 3=demora en el pago por 3 meses , 4=demora en el pago por 4 meses , 5=demora en el pago por 5 meses , 6=demora en el pago por 6 meses , 7=demora en el pago por 7 meses , 8=demora en el pago por 8 meses , 9=demora en el pago por 9 meses o mayor)
# PAY_1: estado de reembolso en septiembre
# PAY_2: estado de reembolso en agosto
# PAY_3: estado de reembolso en julio
# PAY_4: estado de reembolso en junio
# PAY_5: estado de reembolso en mayo
# PAY_6: estado de reembolso en abril
# BILL_AMT1: monto del estado de cuenta en septiembre
# BILL_AMT2: monto del estado de cuenta en agosto
# BILL_AMT3: monto del estado de cuenta en julio
# BILL_AMT4: monto del estado de cuenta en junio
# BILL_AMT5: monto del estado de cuenta en mayo
# BILL_AMT6: monto del estado de cuenta en abril
# PAY_AMT1: monto del pago anterior septiembre
# PAY_AMT2: monto del pago anterior agosto
# PAY_AMT3: monto del pago anterior julio
# PAY_AMT4: monto del pago anterior junio
# PAY_AMT5: monto del pago anterior mayo
# PAY_AMT6: monto del pago anterior abril
# default.payment.next.month: pago por defecto (1=si , 0=no)

# Ingreso de datos
limit_bal = input("limit_bal: ")
sex = input("sex (1=masculino , 2=femenino): ")
education = input("education (1=postgrado , 2=universidad , 3=escuela secundaria , 4=otros , 5=desconocido , 6=desconocido)): ")
marriage = input("marriage (1=casado , 2=soltero, 3=otros): ")
age = input("age: ")

print("\n(-1=pago debidamente , 1=demora en el pago por 1 meses , 2=demora en el pago por 2 meses , 3=demora en el pago por 3 meses , 4=demora en el pago por 4 meses , 5=demora en el pago por 5 meses , 6=demora en el pago por 6 meses , 7=demora en el pago por 7 meses , 8=demora en el pago por 8 meses , 9=demora en el pago por 9 meses o mayor)")

pay_1 = input("pay_1 (estado de reembolso en septiembre): ")
pay_2 = input("pay_2 (estado de reembolso en agosto): ")
pay_3 = input("pay_3 (estado de reembolso en julio): ")
pay_4 = input("pay_4 (estado de reembolso en junio): ")
pay_5 = input("pay_5 (estado de reembolso en mayo): ")
pay_6 = input("pay_6 (estado de reembolso en abril): ")

print("\n(0-∞ = ammount)")

bill_amt1 = input("bill_amt1 (monto del estado de cuenta en septiembre): ")
bill_amt2 = input("bill_amt2 (monto del estado de cuenta en agosto): ")
bill_amt3 = input("bill_amt3 (monto del estado de cuenta en julio): ")
bill_amt4 = input("bill_amt4 (monto del estado de cuenta en junio): ")
bill_amt5 = input("bill_amt5 (monto del estado de cuenta en mayo): ")
bill_amt6 = input("bill_amt6 (monto del estado de cuenta en abril): ")

print("\n(0-∞ = ammount)")

pay_amt1 = input("pay_amt1 (monto del pago anterior septiembre): ")
pay_amt2 = input("pay_amt2 (monto del pago anterior agosto): ")
pay_amt3 = input("pay_amt3 (monto del pago anterior julio): ")
pay_amt4 = input("pay_amt4 (monto del pago anterior junio): ")
pay_amt5 = input("pay_amt5 (monto del pago anterior mayo): ")
pay_amt6 = input("pay_amt6 (monto del pago anterior abril): ")

# Variables
names = []
models = []
results = []
wait = 0
probability = 0.90

# Funciones
def confusionMatrix(confusionMatrixResult, labelsIndex=['Actual: YES','Actual: NO'], labelsColumns=['Predicted: YES','Predicted: NO']):
    df = pd.DataFrame(data=confusionMatrixResult, index=labelsIndex, columns=labelsColumns)
    df.loc['Total'] = df.sum()
    df['Total'] = df.sum(axis=1)
    return df

def individualPrediction(newPrediction):
    data = newPrediction.values.reshape(1, -1)
    data = robustScaler.transform(data)
    predict = logisticRegression.predict_proba(data)[0][1]
    percentage = str(round(predict*100,2))
    if predict >= probability:
        return 'This person will default with a prediction of: ' + percentage + "%"
    else:
        return 'This person will pay with a prediction of: ' + percentage + "%"

# Leer el archivo csv
fileData = pd.read_csv('test.csv', index_col="ID")

# Construccion de los modelos
inputFileData = fileData.drop('default', axis=1)
robustScaler = RobustScaler()
inputRobustScaler = robustScaler.fit_transform(inputFileData)
output = fileData['default']

# random_state = controla el shuffling aplicada a la data antes de separarla
# test_size = representa la proporcion del dataset
# train_test_split = separa la data en entrenamiento y prueba
inputTraining, inputTesting, outputTraining, outputTesting = train_test_split(inputRobustScaler, output, test_size=0.5, random_state=100, stratify=output)

# Modelos
models.append(('Logistic Regression', LogisticRegression()))
models.append(('Decision Tree Classifier', DecisionTreeClassifier()))
models.append(('Naive Bayes Classifier', GaussianNB()))

print('\nModels')
# Loop de los modelos
for name, model in models:

    # KFold = provee indices de entrenamiento para separar la data en sets.
    # random_state = Afecta el orden de los índices, que controla la aleatoriedad de cada fold
    # n_splits = Numero de folds con la que se evalua los modelos con diferentes combinaciones de hiperparametros
    # scoring = Medida del score de la precision del modelo con los datos de entrenamiento
    kfold = KFold(n_splits=10, random_state=7, shuffle=True)

    # Evaluar la puntuacion mediante validacion cruzada
    scoreResult = cross_val_score(model, inputTraining, outputTraining, cv=kfold, scoring='roc_auc')

   # Se Imprime el resultado
    print('\nName: ' + name)
    print('Mean: ' + str(scoreResult.mean()))
    print('Standard deviation: ' + str(scoreResult.std()))

# Espera de n segundos
time.sleep(wait)

# Dataframe para la evaluacion de las metricas
modelResults = pd.DataFrame(index=['Accuracy', 'Precision', 'Recall'], columns=['Logistic Regression', 'Calssification Trees', 'Naive Bayes Classifier'])

# Instancia de los modelos a utilizar para estimar
logisticRegression = LogisticRegression(n_jobs=-1, random_state=15, max_iter = 1000)
calssificationTrees = DecisionTreeClassifier(min_samples_split=30, min_samples_leaf=10, random_state=10)
naiveBayesClassifier = GaussianNB()

# Se utiliza la data de entrenamiento para estimar
logisticRegression.fit(inputTraining, outputTraining)
calssificationTrees.fit(inputTraining, outputTraining)
naiveBayesClassifier.fit(inputTraining, outputTraining)

# Se obtiene las predicciones de los modelos
outputTestingPredictionLogisticRegression = logisticRegression.predict(inputTesting)
outputTestingPredictionCalssificationTrees = calssificationTrees.predict(inputTesting)
outputTestingPredictionNaiveBayesClassifier = naiveBayesClassifier.predict(inputTesting)

# Se evalua el modelo para obtener la precision
modelResults.loc['Accuracy','Logistic Regression'] = accuracy_score(y_pred=outputTestingPredictionLogisticRegression, y_true=outputTesting)
modelResults.loc['Precision','Logistic Regression'] = precision_score(y_pred=outputTestingPredictionLogisticRegression, y_true=outputTesting)
modelResults.loc['Recall','Logistic Regression'] = recall_score(y_pred=outputTestingPredictionLogisticRegression, y_true=outputTesting)
modelResults.loc['Accuracy','Calssification Trees'] = accuracy_score(y_pred=outputTestingPredictionCalssificationTrees, y_true=outputTesting)
modelResults.loc['Precision','Calssification Trees'] = precision_score(y_pred=outputTestingPredictionLogisticRegression, y_true=outputTesting)
modelResults.loc['Recall','Calssification Trees'] = recall_score(y_pred=outputTestingPredictionLogisticRegression, y_true=outputTesting)
modelResults.loc['Accuracy','Naive Bayes Classifier'] = accuracy_score(y_pred=outputTestingPredictionNaiveBayesClassifier, y_true=outputTesting)
modelResults.loc['Precision','Naive Bayes Classifier'] = precision_score(y_pred=outputTestingPredictionLogisticRegression, y_true=outputTesting)
modelResults.loc['Recall','Naive Bayes Classifier'] = recall_score(y_pred=outputTestingPredictionLogisticRegression, y_true=outputTesting)

# Matriz de confusion
LogisticRegressionResult = confusion_matrix(y_pred=outputTestingPredictionLogisticRegression, y_true=outputTesting)
calssificationTreesResult = confusion_matrix(y_pred=outputTestingPredictionCalssificationTrees, y_true=outputTesting)
naiveBayesClassifierResult = confusion_matrix(y_pred=outputTestingPredictionNaiveBayesClassifier, y_true=outputTesting)

# verdaderos positivos = si quedaron en default, y es verdad
# falsos positivos = no quedaron en default y es verdad
# falsos negativos = si quedaron en default, y no es verdad
# verdaderos negativos = no quedaron en default y no es verdad
# Se imprimen los resultados de la matriz de confusion
print("\n\n\nConfusion Matrix")
print('{verdaderos positivos} {falsos positivos}\n{falsos negativos} {verdaderos negativos}')

print("\nLogistic Regression")
print(LogisticRegressionResult)

print("\nClassification Trees")
print(calssificationTreesResult)

print("\nNaive Bayes Classifier")
print(naiveBayesClassifierResult)

# Espera de n segundos
time.sleep(wait)

# Se imprimen los resultados de la matriz de confusion con tablas
print("\n\n\nConfusion Matrix With Tables")

print("\nLogistic Regression")
print(confusionMatrix(LogisticRegressionResult))

print("\nClassification Trees")
print(confusionMatrix(calssificationTreesResult))

print("\nNaive Bayes Classifier")
print(confusionMatrix(naiveBayesClassifierResult))

# Espera de n segundos
time.sleep(wait)

# Se imprimen los resultados del modelo en porcentaje
print("\n\n\nModel Results")
print(modelResults*100)

# Espera de n segundos
time.sleep(wait)

# Data de la nueva prediccion
newPredictionDataDefault = OrderedDict([('limit_bal', 10000),('sex', 1),('education', 1),('marriage', 1),('age', 30),('pay_1', 9),('pay_2', 9),('pay_3', 9),('pay_4', 9),('pay_5', 9), ('pay_6', 0),('bill_amt1', 10),('bill_amt2', 20 ),('bill_amt3', 30 ),('bill_amt4', 40 ),('bill_amt5', 50 ),('bill_amt6', 60 ), ('pay_amt1', 100 ),('pay_amt2', 200 ),('pay_amt3', 300 ),('pay_amt4', 400 ),('pay_amt5', 500 ), ('pay_amt6', 600 )])
newPredictionDataPay = OrderedDict([('limit_bal', 20000),('sex', 2),('education', 2),('marriage', 2),('age', 30),('pay_1', -1),('pay_2', -1),('pay_3', -1),('pay_4', -1),('pay_5', -1), ('pay_6', 0),('bill_amt1', 100),('bill_amt2', 200 ),('bill_amt3', 300 ),('bill_amt4', 400 ),('bill_amt5', 500 ),('bill_amt6', 600 ), ('pay_amt1', 1000 ),('pay_amt2', 2000 ),('pay_amt3', 3000 ),('pay_amt4', 4000 ),('pay_amt5', 5000 ), ('pay_amt6', 6000 )])
newPredictionDataEntry= OrderedDict([('limit_bal', limit_bal),('sex', sex),('education', education),('marriage', marriage),('age', age),('pay_1', pay_1),('pay_2', pay_2),('pay_3', pay_3),('pay_4', pay_4),('pay_5', pay_5), ('pay_6', pay_6),('bill_amt1', bill_amt1),('bill_amt2', bill_amt2 ),('bill_amt3', bill_amt3 ),('bill_amt4', bill_amt4 ),('bill_amt5', bill_amt5 ),('bill_amt6', bill_amt6 ), ('pay_amt1', pay_amt1 ),('pay_amt2', pay_amt2 ),('pay_amt3', pay_amt3 ),('pay_amt4', pay_amt4 ),('pay_amt5', pay_amt5 ), ('pay_amt6', pay_amt6 )])
# Nueva prediccion
newPredictionDefault = pd.Series(newPredictionDataDefault)
newPredictionPay = pd.Series(newPredictionDataPay)
newPredictionEntry = pd.Series(newPredictionDataEntry)

# Se imprime la data
print("\n\n\nNew Prediction Data")
print("\nNew Prediction Data Default")
print(newPredictionDataDefault)
print("\nNew Prediction Data Pay")
print(newPredictionDataPay)
print("\nNew Prediction Entry")
print(newPredictionDataEntry)

# Espera de n segundos
time.sleep(wait)

# Se imprime el resultado
print("\n\n\nNew Prediction Result")
print("\nNew Prediction Default")
print(individualPrediction(newPredictionDefault))
print("\nNew Prediction Pay")
print(individualPrediction(newPredictionPay))
print("\nNew Prediction Entry")
print(individualPrediction(newPredictionEntry))
print("\n")

# Espera de 60 segundos
time.sleep(60)