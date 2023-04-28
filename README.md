# **Bildklassifikation - 😺 vs. 🐶**

Im Laufe dieses Demo wird ein **Machine-Learnin-Modell** entwickelt, welches in der Lage ist zwischen Bildern von **Katzen und Hunden** zu unterscheiden. Das Modell wird auf einem **neuralen Netzwerk** basieren und wird mit einer umfangreichen Datensammlung **überwacht** trainiert. 
![cats and dogs](/header.gif)

## Anforderungen
Für dieses Demo wird nicht vorausgesetzt, da wir in [Google Colab](https://colab.research.google.com/) arbeiten werden.

## Einleitung 

### Definition Machine-Learning

> Machine Learning ist ein Teilbereich der künstlichen Intelligenz. Mithilfe des maschinellen Lernens werden IT-Systeme in die Lage versetzt, auf Basis vorhandener 
> Datenbestände und Algorithmen Muster und Gesetzmäßigkeiten zu erkennen und Lösungen zu entwickeln. Es wird quasi künstliches Wissen aus Erfahrungen generiert. Die aus  > den Daten gewonnenen Erkenntnisse lassen sich verallgemeinern und für neue Problemlösungen oder für die Analyse von bisher unbekannten Daten verwenden.
>
> <cite>https://www.bigdata-insider.de/was-ist-machine-learning-a-592092/</cite>

### Definition neurales Netzwerk 

> Das Künstliche Neuronale Netz (KNN) ist bis zu einem gewissen Grad dem Aufbau des biologischen Gehirns nachempfunden. Es besteht aus einem abstrahierten Modell 
> miteinander verbundener **Neuronen**, durch deren spezielle Anordnung und Verknüpfung sich Anwendungsprobleme aus verschiedenen Bereichen computerbasiert lösen
> lassen. 
> Bevor ein Neuronales Netzwerk für die vorgesehen Problemstellung oder Aufgabe verwendbar ist, muss es zunächst trainiert werden. Anhand von vorgegebenem Lernmaterial > und Lernregeln gewichtet das Neuronale Netz die Verbindungen der Neuronen, bis es eine bestimmte „Intelligenz“ entwickelt hat. Die Lernregeln geben vor, wie das
> Lernmaterial das Neuronale Netz verändert. Grundsätzlich kann zwischen dem **überwachten Lernen** und dem **unüberwachten Lernen** unterschieden werden. Beim
> überwachten Lernen wird ein konkretes Ergebnis für die unterschiedlichen Eingabemöglichkeiten vorgegeben. Anhand des ständigen Vergleichs zwischen Soll- und Ist-
> Ergebnis lernt das Netz die Neuronen passend zu verknüpfen. Unbeaufsichtigte Lernen gibt kein Ergebnis vor. 
>
> <cite>https://www.bigdata-insider.de/was-ist-ein-neuronales-netz-a-686185/</cite>

## Datensatz

https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip ist ein öffentlich verfügbares Datenset, das vom Machine Learning Education Team von Google bereitgestellt wird. 
Der Datensatz enthält **3000 Bilder** von Hunden und Katzen mit den **Labels**:
- 1 = Hund
- 0 = Katze 

## Wie funktionierts?
1. Daten vorbereiten
2. Model definieren
3. Model trainieren
4. Model validieren 

### Daten vorbereiten
Grundsätzlich werden bei der Ersellung eines Bildklassifizierungsmodell drei seperate Datensätze benötigt:

1. Trainingsdaten
2. Validierungsdaten, welche zur Validierung während des Trainings dienen
3. Testdaten, um das fetig trainierte Model zu testen

Es muss sichergestellt werden, dass alle Bilder richtig kategorisiert wurden. Die Bilder müssen alle auf die gleiche Größe gebracht werden. Es ist üblich Augmenterungstechniken auf die Bilder im Trainingsdatensatz anzuwenden. 

### Model definieren

            
## Autor
Elisabeth Wadler

