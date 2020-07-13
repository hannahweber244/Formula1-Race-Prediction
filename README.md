# Formula1-Race-Prediction
Machine Learning Projekt für das Fach Data Exploration Project an der DHBW Mannheim, Team F1

## 1. Projektziel:
Ziel dieses Projekts ist es ein (oder mehrere) Modelle zur Verfügung zu stellen, welche die Top X Platzierungen eines beliebiegen Formel 1 Rennens vorhersagen. 

## 2. Projektaufbau:
### 2.1 Modelle:
1. Lineare Regression: <br>
Es ist mit Hilfe der sklearn Klassen LinearRegression(), Lasso() und Ridge() eine Lineare Regression ohne-, mit L1- und L2-Regularisierung implementiert worden. Ziel der Regularisierung ist es die sehr kleinen Koeffizienten der kategorischen (one.hot-encodeden) Attribute zu limitieren. 

2. Neuronales Netz: <br>
Um die Nichtlinearität in den Daten zu lernen wurde sich für ein Neuronales Netz entschieden. Dieses wird mit Hilfe von pytorch implementiert

### 2.2 Code:
Der Code ist in zwei Formaten verfügbar:
1. Jupyter Notebook
2. py-Dateien
Beide Formate könne gleichwertig verwendet werden, die Jupyter Notebooks bieten jedoch eine bessere visuelle Unterstützung und zusätzliche Informationen zum Vorgehen. Code für die Lineare Regression ist nur in einem Jupyter Notebook verfügbar.

### 2.3 Ausführung des Codes:
...



## 3. Team Mitglieder:
Julian Greil (3451503) <br>
Florian Köhler (4810569) <br>
Hannah Weber (3143200) <br>
Manuel Zeh (6061471) 
