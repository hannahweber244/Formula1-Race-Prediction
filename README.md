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
Der Code wurde in Jupyter Notebooks entwickelt und wird aus diesem Grund in diesem Format bereitgestellt. 

### 2.4 Codeinhalt:

- Jupyter Notebooks:
  - Datenaufbereitung.ipynb: Datenverarbeitung, -aufbereitung und Export
  - LineareRegression Final.ipynb: Lineare Regression und Regularisierung mit Ergebnissen
  - NeuronalesNetz.ipynb: Hyperparameteroptimierung, Neuronales Netz, Training und finales Testen
  
- dependencies.txt: Package Dependencies und Requirements
- Final_F1.pt: Pytorch Datei, die das finale, vortrainierte Neuronale Netz enthält 
- kaggle_data: Ordner, welcher die Rohdaten von Kaggle enthält
- sliced_data: Ordner, welcher Formel-1-Rennen bis zur 50% Marke enthält

### 2.5 Ausführung des Codes:
Bevor der Code ausgeführt wird, sollte sichergestellt werden, dass alle Requirements aus der dependencies.txt Datei (siehe auch 2.2 Requirements) installiert sind. Besonders wichtig ist es, für die lokale Entwicklungsumgebung das Package pytorch mit den richtigen Spezifikationen zu installieren. Eine Anleitung, wie die Installation durchgeführt werden sollte ist unter https://pytorch.org/ zu finden. Im Code selbst ist es egal, welche pytorch Version und ob CUDA verwendet wird.



**Datenaufbereitung.ipynb:**<br>
Das Notebook Datenaufbereitung.ipynb muss im gleichen Ordner liegen, wie der Ordner "kaggle_data".<br>
Das Notebook kann dann ohne weiteres von oben nach unten durchgeladen werden.<br>
Output des Notebooks: Ordner sliced_data, wie auch in Git bereitgestellt <br>
Rechendauer ~ 30/45min

**LineareRegression.ipynb:**<br>
Das Notebook muss im gleichen Verzeichnis liegen, wie sliced_data, damit aus diesem die CSV Dateien geladen werden können. Bei der Ausführung des Notebooks müssen keine Besonderheiten beachtet werden. Alle relevanten Informationen sind im Notebook zu finden.

**NeuronalesNetz.ipynb:**<br>
Das Notebook muss im gleichen Verzeichnis liegen, wie sliced_data, damit aus diesem die CSV Dateien geladen werden können. Das Notebook kann in zwei Varianten ausgeführt werden (siehe 1. und 2. Möglichkeit). In jedem Fall sollte **auf die angegebenen Ausführzeiten geachtet werden**, da diese insgesamt ~17h betragen können!

   **1. Möglichkeit:** 
   Der Code soll in Google Colab ausgeführt werden. Packages müssen in der Umgebung eventuell installiert werden (pytorch, pandasql, etc.). Die Installation noch nicht in Umgebung installierter Packages funktioniert wie folgt:<br>
        ```
            !pip install [packagename]
        ```<br>
  Ein Beispiel kann hier <br>
        ```
        !pip3 install torch==1.2.0+cu92 torchvision==0.4.0+cu92 -f    https://download.pytorch.org/whl/torch_stable.html
        ```<br>
    sein, um pytorch für Google Colab zu installieren. Wenn Google Colab verwendet wird, müssen Daten über einen Umweg in das Notebook geladen werden, da sie zunächst auf Google Colab hochgeladen werden müssen. Die entsprechende Zelle ist im Notebook gekennzeichnet und mit allen relevanten HInweisen versehen.

  **2. Möglichkeit:** <br>
  Der Code wird lokal ausgeführt. Bei dieser Variante muss zur Ausführung des Notebooks nichts weiter beachtet werden. Bei der Ausführung sollte nur die Google Colab Zelle für den Datenupload nicht mit ausgeführt werden. Anweisungen im Code folgen!


### 2.6 Daten:
Die Daten stammen von kaggle (https://www.kaggle.com/cjgdev/formula-1-race-data-19502017) und liegen als Rohdaten in dem Ordner kaggle_data vor. Mit Hilfe des entsprechenden Jupyter Notebooks (Datenaufbereitung.ipynb) können diese aufbereitet und für die Modelle vorbereitet werden. Anstelle dessen können auch die schon verarbeiteten Daten aus dem Ordner sliced_data verwendet werden, die extra für diesen Zweck vorbereitet worden sind. 

## 3. Team Mitglieder:
Julian Greil (3451503) <br>
Florian Köhler (4810569) <br>
Hannah Weber (3143200) <br>
Manuel Zeh (6061471) 
