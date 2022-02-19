#Librerias requeridas para el proyecto
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
from sklearn.metrics import  classification_report, confusion_matrix



#Cargamos el dataset.
dataset = pd.read_csv('data.csv')
pd.set_option('display.max_columns', None) #Le indicamos que muestre todas las columnas.

#Analizamos el dataset.
def dataSetAnalysis(df):
    #Mostralos las primeras filas.
    print("\n*************************  Head del dataset  *************************\n")
    print(df.head())

    
    #Mostramos las columnas y sus nombres.
    print("\n*************************  Atributos del dataset  *************************\n")
    print(df.columns.values)
    print("\n")

    
    #Descripcion de los atributos numericos del dataset:
    #Cantidad
    #Promedio
    #Desviacion estandar
    #Minimo
    #Cuartiles
    #Maximo
    print("\n*************************  Dataset atributos numericos  *************************\n")
    print(df.describe())
    print("=" * 60)
    print("=" * 60)
    print("=" * 60)
    

#Mostramos datos.
dataSetAnalysis(dataset)
diagnosis_all = list(dataset.shape)[0]
diagnosis_categories = list(dataset['diagnosis'].value_counts())

print("\n\nLos datos tienen {} diagnosticos, {} benignos y {} malignos.".format(diagnosis_all,
                                                                                 diagnosis_categories[0], 
                                                                                 diagnosis_categories[1]))
print("\n\n Benignos representan al",  float(diagnosis_categories[0]*100/diagnosis_all) , " %")
print("Malignos representan al" , float(diagnosis_categories[1]*100/diagnosis_all) , " %\n\n")

#Preparamos el dataset para poder analizar los datos. Sacamos las columnas que no son relevantes.
data = pd.read_csv('data.csv')
data = data.drop('id',axis=1)
data = data.drop('Unnamed: 32',axis=1)
# Mapeamos benignos a 0 y malignos a 1.
data['diagnosis'] = data['diagnosis'].map({'M':1,'B':0})
data.describe()


#Grafico de barra diagnostico vs cantidad de c/tipo.
sns.countplot(dataset['diagnosis'], label="Count", palette=sns.color_palette(['#006400', '#DC143C']),
              order=pd.value_counts(dataset['diagnosis']).iloc[:17].index)
plt.show()


#Buscamos correlacion entre variables del dataset. Queremos ver la fuerza de la relacion entre ellas.
breast_cancer = data
breast_cancer_corr = breast_cancer.corr()
print(breast_cancer_corr)


#Damos forma al Heatmap.
plt.figure(figsize=(16, 9))
sns.heatmap(breast_cancer_corr, vmax=1, annot=False, cmap="Blues", fmt='d', cbar=True, yticklabels=breast_cancer.columns,
             linewidth=2, xticklabels=breast_cancer.columns)
plt.show()

#Graficamos dos variables con alta correlacion.  radius_worst vs perimeter_worst. Obtenemos un analisis de la fuerza de esta relacion.
#Tambien podemos ver donde se ubica la mayor parte de nuestra muestra y la recta originada por la regresion lineal.
sns.jointplot("radius_worst", "perimeter_worst", data=breast_cancer, kind="reg",
              space=0, color="#4185FB", height=5, ratio=3)
plt.show()

#Otra manera de visualizar
sns.pairplot(dataset, vars=["radius_worst", "perimeter_worst"],
             palette=sns.color_palette(['#5bbbeb', '#22324e']), hue='diagnosis', height=3)
plt.show()

#Graficamos dos variables con baja correlacion.  radius_mean vs fractal_dimension_mean. Obtenemos un analisis de la fuerza de esta relacion.
#Tambien podemos ver donde se ubica la mayor parte de nuestra muestra y la recta originada por la regresion lineal.
sns.jointplot("radius_mean", "fractal_dimension_mean", data=breast_cancer, kind="reg",
              space=0, color="#4185FB", height=5, ratio=3)
plt.show()

#Otra manera de visualizar
sns.pairplot(dataset, vars=["radius_mean", "fractal_dimension_mean"],
             palette=sns.color_palette(['#5bbbeb', '#22324e']), hue='diagnosis', height=3)
plt.show()


dataMalignant=dataset[dataset['diagnosis'] == 1]
dataBenign=dataset[dataset['diagnosis'] == 0]

#Definimos variables dependiente e independiente.
X = dataset.iloc[:,2:32] #La variable independiente sera la fila.
y = dataset.iloc[:,1] #La variable dependiente es la columna 1, quien indica M o B
labelencoder_Y = LabelEncoder()
y = labelencoder_Y.fit_transform(y)

#Separamos los datos de entrenamiento de los datos de prueba.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Busqueda de lo hiper parametros. Lo comentamos porque luego lo cargamos.
'''
 from keras.wrappers.scikit_learn import KerasClassifier
 from sklearn.model_selection import GridSearchCV
 from keras.models import Sequential
 from keras.layers import Dense

 def build_classifier(optimizer):
     classifier = Sequential()
     classifier.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu', input_dim = 30))
     classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu'))
     classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
     classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
     classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
     return classifier
 classifier = KerasClassifier(build_fn = build_classifier)
 parameters = {'batch_size': [1, 5],
               'epochs': [100, 120],
               'optimizer': ['adam', 'rmsprop']}
 grid_search = GridSearchCV(estimator = classifier,
                            param_grid = parameters,
                            scoring = 'accuracy',
                            cv = 10)
 grid_search = grid_search.fit(X_train, y_train)

 best_parameters = grid_search.best_params_
 best_accuracy = grid_search.best_score_
 print("best_parameters: ")
 print(best_parameters)
 print("\nbest_accuracy: ")
 print(best_accuracy)
'''
#Inicializamos el modelo con los parametros encontrados.
#Tambien lo cargamos para ahorrar tiempo. Queda comentado.
'''
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential() # Initialising the ANN

classifier.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu', input_dim = 30))
classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

classifier.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit(X_train, y_train, batch_size = 1, epochs = 100)
'''
#Guardamos el modelo.
#classifier.save('breast_cancer_model.h5')

#Cargamos el modelo ya entrenado.
classifier = load_model('breast_cancer_model.h5')


#Dejamos de usar los datos de entrenamiento y empezamos a usar los datos de prueba.
y_pred = classifier.predict(X_test)
y_pred = [ 1 if y>=0.5 else 0 for y in y_pred ]

#Graficamos los resultados en una matriz de confusion
valores = {'y_test':y_test,
        'y_pred': y_pred}

df = pd.DataFrame(valores,columns=['y_test','y_pred'])
confu_ma = pd.crosstab(df['y_test'], df['y_pred'], rownames=['Obtenidos'], colnames=['Predecidos'])
sns.heatmap(confu_ma, annot=True, linewidths=1, xticklabels=(['Malignos', 'Benignos']), yticklabels=(['Malignos', 'Benignos']))
plt.show()

#Mostramos el clasification report. A continuacion dejamos comentado que significa cada parametro.
'''
Precision es el radio tp / (tp + fp). Habilidad del clasificador para no predecir como positiva una muestra negativa.

Recall es el radio tp / (tp + fn). Habilidad del clasificador para encontrar todas las muestras positivas.

F-beta es una media armonica ponderada de la precision y el recall, mejores valores cercanos a 1 y peores a 0.

Soporte es numero de ocurrencias de cada clase en valores reales.
'''
print("\n\n\n **************** Los resultados de la red neuronal son: **********************\n\n\n")
print(classification_report(y_test, y_pred))

#Calculamos la precision final del modelo segun los datos obtenidos en la matriz de confusion.
cm = confusion_matrix(y_test,y_pred)
accuracy = (cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])

#Si calculamos de la matriz de confusion: Buenas predicciones/ Predicciones totales.
print("Precision: "+ str(accuracy*100)+"%")
