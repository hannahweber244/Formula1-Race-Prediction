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

### 2.2 Requirements:
Die folgende Tabelle enthält alle wichtigen Packages und Dependencies für dieses Projekt. Die Ausnahme ist pytorch, was je nach Systemspezifikation anders heruntergeladen werden muss: siehe https://pytorch.org/
| Package       | Version        |
| ------------- |:-------------:|
| cycler      | 0.10.0 |
| joblib      | 0.16.0   |
| kiwisolver | 1.2.0 |
| matplotlib   | 3.2.2 |
| numpy     | 1.19.0   |
| pandas | 1.0.5     |
| pandasql   | 0.7.3 |
| pyparsing     | 2.4.7    |
| python-dateutil | 2.8.1     | 
| pytz   | 2020.1 | 
| scikit-learn     | 0.23.1     | 
| scipy | 1.5.1      | 
| seaborn      | 0.10.0 |
| six     | 1.15.0    | 
| sklearn | 0.0    |  
|SQLAlchemy | 1.3.18 |
| threadpoolctl | 2.1.0 |

### 2.3 Codeformat:
Der Code ist in zwei Formaten verfügbar:
1. Jupyter Notebook
2. py-Dateien
<br>Die Jupyter Notebooks bieten eine bessere Unterstützung, als die einfachen .py-Dateien. 

### 2.4 Codeinhalt:

- Jupyter Notebooks:
  - Datenaufbereitung.ipynb: Datenverarbeitung, -aufbereitung und Export
  - LineareRegression Final.ipynb: Lineare Regression und Regularisierung mit Ergebnissen
  - NeuronalesNetz.ipynb: Hyperparameteroptimierung, Neuronales Netz, Training und finales Testen
  
- .py Dateien:
  - main.py: Startet 

### 2.5 Ausführung des Codes:
**1. Möglichkeit:** Der Code soll in Google Colab ausgeführt werden:<br>
   Installation noch nicht in Umgebung installierter Packages:<br>
    ```
        !pip install [packagename]
    ```<br>
    Weitere Hinweise zum Vorgehen finden sich im Code selbst!

**2. Möglichkeit:** Der Code wird lokal ausgeführt<br>
Anweisungen im Code folgen!

**Datenaufbereitung**<br>
Die Daten aus dem Ordner "kaggle data" müssen im selben Ordner wie das Notebook bereit liegen und müssen dort die einzigen verfügbaren .csv Dateien sein.<br>
Das Notebook kann von oben nach unten durchgeladen werden.<br>
Output des Notebooks: Ordner sliced_data, wie auch in Git bereitgestellt <br>
Rechendauer ~ 30/45min


### 2.6 Daten:
Die Daten können entweder von kaggle direkt heruntergeladen werden (https://www.kaggle.com/cjgdev/formula-1-race-data-19502017) und dann mit Hilfe des entsprechenden Jupyter Notebooks (Datenaufbereitung.ipynb) vorbereitet werden, oder die schon verarbeiteten Daten können aus dem Ordner sliced_data verwendet werden. 
## 3. Team Mitglieder:
Julian Greil (3451503) <br>
Florian Köhler (4810569) <br>
Hannah Weber (3143200) <br>
Manuel Zeh (6061471) 
