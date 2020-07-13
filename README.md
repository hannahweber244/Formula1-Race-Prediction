# Formula1-Race-Prediction
Machine Learning Projekt für das Fach Data Exploration Project an der DHBW Mannheim, Team F1

## 1. Projektziel:
Ziel dieses Projekts ist es ein (oder mehrere) Modelle zur Verfügung zu stellen, welche die Top 10 Platzierungen eines beliebiegen Formel 1 Rennens vorhersagen. 

## 2. Projektaufbau:
### 2.1 Modelle:
1. Lineare Regression: <br>
Es ist mit Hilfe der sklearn Klassen LinearRegression(), Lasso() und Ridge() eine Lineare Regression ohne-, mit L1- und L2-Regularisierung implementiert worden. Ziel der Regularisierung ist es die sehr kleinen Koeffizienten der kategorischen (one-hot-encodeden) Attribute zu limitieren. 

2. Neuronales Netz: <br>
Um die Nichtlinearität in den Daten zu lernen wurde sich für ein Neuronales Netz entschieden. Dieses wird mit Hilfe von pytorch implementiert. Im Zuge der Hyperparameteroptimierung werden Neuronale Netze dynamisch erzeugt und miteinander verglichen.

### 2.2 Codeformat:
Der Code ist in zwei Formaten verfügbar:
1. Jupyter Notebook
2. py-Dateien
<br>Beide Formate könne gleichwertig verwendet werden, die Jupyter Notebooks bieten jedoch eine bessere visuelle Unterstützung und zusätzliche Informationen zum Vorgehen. Der Code für die Lineare Regression ist nur in einem Jupyter Notebook verfügbar.

### 2.3 Codeinhalt:

- Jupyter Notebooks:
  - Datenaufbereitung.ipynb --> Datenverarbeitung, -aufbereitung und Export
  - LineareRegression Final.ipynb

### 2.4 Ausführung des Codes:
#### Der Code soll in Google Colab ausgeführt werden:<br>
  Die load_data() Funktion kann nicht aufgerufen werden, stattdessen wird folgender Code verwendet:<br>
    ```
    from google.colab import files
    
    uploaded = files.upload()
    ```


### 2.5 Daten:
Die Daten können entweder von kaggle direkt heruntergeladen werden (https://www.kaggle.com/cjgdev/formula-1-race-data-19502017) und dann mit Hilfe des entsprechenden Jupyter Notebooks (Datenaufbereitung.ipynb) vorbereitet werden, oder die schon verarbeiteten Daten können aus dem Ordner sliced_data verwendet werden. 
## 3. Team Mitglieder:
Julian Greil (3451503) <br>
Florian Köhler (4810569) <br>
Hannah Weber (3143200) <br>
Manuel Zeh (6061471) 
