# **Bildklassifikation - 😺 vs. 🐶**

Im Laufe dieser Demo wird ein **Machine-Learning-Modell** entwickelt, welches in der Lage ist zwischen Bildern von **Katzen und Hunden** zu unterscheiden. Das Modell wird auf einem **neuralen Netzwerk** basieren und wird mit einer umfangreichen Datensammlung **überwacht,** trainiert. 
![cats and dogs](/header.gif)
Quelle: https://raihanrnj.medium.com/deep-learning-simple-image-classification-using-convolutional-neural-network-dog-and-cat-8c99aef29e8

## Anforderungen
Für diese Demo wird nichts vorausgesetzt, da wir in [Google Colab](https://colab.research.google.com/) arbeiten werden.

## Einleitung 

### Definition Machine Learning

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

https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip ist ein öffentlich verfügbares Datenset, das vom "Machine Learning Education" Team von Google bereitgestellt wird. 
Der Datensatz enthält **3000 Bilder** von Hunden und Katzen mit den **Labels**:
- 1 = Hund
- 0 = Katze 

## Wie funktioniert es?
1. [Daten vorbereiten](#Daten-vorbereiten)
2. [Modell definieren](#Modell-definieren)
3. [Modell trainieren](#Modell-trainieren)
4. [Modell validieren](#Modell-validieren)

### Daten vorbereiten
Grundsätzlich werden bei der Erstellung eines Bildklassifizierungsmodelles drei separate Datensätze benötigt:

1. Trainingsdaten
2. Validierungsdaten, welche zur Validierung während des Trainings dienen
3. Testdaten, um das fertig trainierte Model zu testen

Es muss sichergestellt werden, dass alle Bilder **richtig kategorisiert** wurden. Die Bilder müssen alle auf die gleiche Größe skaliert werden. Es ist üblich **Augmenterungstechniken** (drehen, zoomen, spiegeln, ...) auf die Bilder im Trainingsdatensatz anzuwenden. So kann ein kleiner Datensatz künstlicher vergrößert werden.

### Modell definieren
In TensorFlow kann mithilfe der Keras-API ein mehrschichtiges neuronales Netz/Modell definiert werden. In unserem Beispiel wird damit ein sequenzielles neuronales Netz aufgebaut. Die Ausgabe einer jeden Schicht dient dabei als Eingabe der nächsten Schicht. Die Netztopologie kann somit einfach von oben nach unten gelesen werden. In diesem wird auch der Kompelierungsprozess definiert. 

![model](/model_definition.png)  

#### Verwendete Schichten
- **Convolutional-Schicht / Faltungsebenen** \
Kann in den Eingabedaten einzelne **Merkmale erkennen** und extrahieren.  Dabei werden **kleine Filterkerne** über das Eingangsbild bewegt, um Merkmale zu extrahieren. Jeder Filterkern lernt verschiedene Merkmale zu erkennen, wie z. B. Kanten, Texturen oder bestimmte Formen. Es können die Parameter Anzahl der Filterkerne, Größe der Filterkerne, Aktivierungsfunktion und erwartetes Eingabeformat definiert werden

- **Pooling-Schicht / Subsampling-Schich** \
Verdichtet und reduziert die Auflösung der erkannten Merkmale. Das Pooling **verwirft überflüssige Informationen** und behält gleichzeitig wichtige Merkmale. Die MaxPooling2D-Schicht verwendet die Max-Pooling-Operation, bei der der maximale Wert innerhalb des Polling-Fensters ausgewählt wird. Dieser Wert wird dann in der Ausgabe beibehalten, während andere Werte verworfen werden. Dadurch werden wichtige Merkmale mit maximaler Aktivierung beibehalten.

- **Dropout-Schicht** \
Dient zur **Vermeidung von "Overfitting"**. Die Schicht gibt and, dass während eines Trainingsschrittes ein zufälliger Teil der Neuronen ignoriert wird. (Bsp. Dropout(0.3) = 30 % der Neuronen werden ignoriert/verworfen) dadurch wird verhindert, dass sich Neuronen im Modell zu stark auf bestimmte Eingaben oder Merkmale spezialisieren.

- **Flatten**
Der Flatten-Layer in einem sequenziellen Modell in Keras wird verwendet, um die **Dimensionalität** der Daten vor dem Übergang zu den dichten (fully connected) Schichten zu **reduzieren**. Dabei nimmt er die Ausgabe der vorherigen Schicht und wandelt sie in einen **eindimensionalen Vektor** um. Die Hauptaufgabe der Flatten-Schicht ist es, die Daten in ein Format zu bringen, das von der nachfolgenden (Dense-Schicht) verarbeitet werden kann. 

- **Dense**
Enthält vollständig verbundene Neuronen/Units. Es ist die **grundlegende Schicht für die Neuronenverbindung** in einem neuralem Netzwerk. Der Dense Layer empfängt einen eindimensionalen Vektor und wendet eine **lineare Transformation** auf ihn an, gefolgt von einer **Aktivierungsfunktion**. Die Dense-Layer ermöglichen es dem Modell, **komplexe, nicht lineare Zusammenhänge zwischen den Eingabedaten** zu lernen. Durch die Kombination der linearen Transformation und der Aktivierungsfunktion kann der Dense-Layer komplexe Muster erkennen und extrahieren. Es können verschiedene Parameter eingestellt werden: Anzahl der Neuronen im Layer, die Aktivierungsfunktion. Die **Sigmoid-Aktivierungsfunktion** wird für die binäre Klassifikation verwendet. Da die Sigmoid-Funktion den Bereich zwischen 0 und 1 abdeckt, kann sie verwendet werden, um die Wahrscheinlichkeit zu berechnen, dass eine Eingabe zu einer bestimmten Klasse gehört. 

### Modell trainieren 
Nachdem das Modell definiert wurde, kann es trainiert werden. Dazu wurde die model.fit Methode verwendet. Der Methode werden die Trainings- und Validierungsdaten übergeben. Zudem werden die Anzahl der Epochen als auch Schritte pro Epoche festgelegt. 

### Modell validieren 
Um Aussagen über die **Qualität** des Modells treffen zu können, gibt es einige Validierungsmethoden. Die Modellvalidierung ist ein wichtiger Schritt des Entwicklungsprozesses von Systemen des maschinellen Lernens, da sie dazu beiträgt, sicherzustellen, dass das Modell die beabsichtigte Leistung erbringt und ungesehene Daten verarbeiten kann.

#### Confusion Matrix
Ist ein Werkzeug zur Bewertung der Leistung eines Klassifikationsmodells. Sie bietet eine detaillierte Übersicht über die Anzahl der **richtigen und falschen Vorhersagen** des Modells für jede Klasse.

![ConfusionMatrix](/ConfusionMatrix.png) 

#### Model.evaluate()
Gibt loss und accuracy für den gegebenen Datensatz zurück.\
**Loss (Verlust)**: Der Loss-Wert ist ein Maß für den Unterschied zwischen den vorhergesagten Wahrscheinlichkeitsverteilungen für die verschiedenen Klassen und den tatsächlichen Labelwerten. \
**Accuracy (Genauigkeit)**: Die Genauigkeit ist ein Metrikwert, der angibt, wie gut das Modell bei der Vorhersage der richtigen Klasse ist. Sie wird oft als Prozentsatz angegeben und gibt an, wie viele Beispiele in der Testdatenmenge korrekt klassifiziert wurden. Eine Genauigkeit von 1,0 bedeutet, dass das Modell alle Beispiele richtig klassifiziert hat, während eine Genauigkeit von 0,5 bedeutet, dass das Modell zufällige Vorhersagen macht.

#### f1-score
Der F1-Score ist eine Metrik zur Bewertung der Leistung eines Klassifikationsmodells, insbesondere bei **ungleich verteilten Klassen** oder wenn sowohl Präzision als auch Recall wichtig sind.

### Heatmap
Eine Heatmap ist eine visuelle Darstellung von Daten, die in Form einer Farbskala dargestellt werden. 
Sie sind nützlich, um bestimmte **Muster und Zusammenhänge in den Daten** zu erkennen. Zum Beispiel kann man sie nutzen, um zu sehen, welche Bereiche in einem Bild am meisten Aktivität aufweisen, also für das Model am relevantesten für die Klassifizierung sind. 

![Heatmap](/Heatmap.png) 

## Autor
Elisabeth Wadler

## Referenzen
https://www.bigdata-insider.de/was-ist-machine-learning-a-592092/
https://www.bigdata-insider.de/was-ist-ein-convolutional-neural-network-a-801246/#:~:text=Die%20Convolutional%2DSchicht%20ist%20die,erfolgt%20in%20Form%20einer%20Matrix.
https://www.bigdata-insider.de/was-ist-ein-neuronales-netz-a-686185/
https://keras.io/
https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-to-classify-photos-of-dogs-and-cats/
Deep Learning with Python by Francois Chollet
