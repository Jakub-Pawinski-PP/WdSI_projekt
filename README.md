# Wprowadzenie
Projekt na laboratorium Wprowadzenia do Sztucznej Inteligencji: system wykrywający znaki drogowe na zdjęciach.

# Objaśnienie koncepcji
Projekt powstał z użyciem biblioteki Detecto, która to pozwala stworzyć sieć neuronową na podstawie bazy udostępnionych fotografii do treningu. Dokumentacja biblioteki znajduje się pod poniższym linkiem:

https://detecto.readthedocs.io/en/latest/

Prezentacja projektu z wykorzystaniem możliwości Google Colaboratory została przedstawiona w pliku road_signs_detection.ipynb:

https://github.com/Jakub-Pawinski-PP/WdSI_projekt/blob/main/road_signs_detection.ipynb

Omówienie koncepcji bazuje na kodzie pochodzącym z main.py, natomiast wyniki poszczególnych funkcji pochodzą z powyższego. Różnice występują jedynie w ścieżkach projektowych.

# Wymagane biblioteki

Pierwszymi krokami jest zaimportowanie wymaganych bibliotek oraz podanie uniwersalnych ścieżek projektu. Odbywają się pod poniższymi listingami:

```
import os
from detecto import core, utils, visualize
import numpy as np
```
Są to odpowiednio biblioteka os (do wyznaczenia ścieżek projektowych), funkcje z biblioteki Detecto używane w procesie uczenia oraz weryfikacji oraz numpy w projekcie filtru.

Jeżeli biblioteka Detecto nie jest zainstalowana w systemie, należy posłużyć się komendą:

```
pip3 install detecto
```

# Ścieżki projektowe

Zgodnie z wytycznymi dotyczącymi podziału katalogów zostały one podzielone na trzy foldery: folder treningowy, folder testowy oraz folder z kodem. Po pobraniu repozytorium w obecnej strukturze ścieżki powinny zostać zachowane.

Ścieżka do całego projektu:

```
#sciezka calego projektu
project_path = os.path.dirname(os.getcwd())
```

Ścieżka do zbioru treningowego:

```
#sciezka zbioru treningowego
train_path = os.path.join(project_path, 'train')
train_annotations_path = os.path.join(train_path, 'annotations')
train_images_path = os.path.join(train_path, 'images')
train_annotations_files = os.listdir(train_annotations_path)
train_images_files = os.listdir(train_images_path)
```

Ścieżka do zbioru testowego:

```
#sciezka zbioru testowego
test_path = os.path.join(project_path, 'test')
test_annotations_path = os.path.join(test_path, 'annotations')
test_images_path = os.path.join(test_path, 'images')
test_annotations_files = os.listdir(test_annotations_path)
test_images_files = os.listdir(test_images_path)
```

Folder treningowy oraz testowy zawierają dwa podfoldery. W pierwszym znajdują się pliki .xml, które zawierają dane o obiekcie występującym na zdjęciu (typ oraz współrzędne prostokąta je obejmującego), natomiast w drugim same już zdjęcia w formatach .png. Pliki w formacie .xml są de facto wykorzystywane jedynie w procesie uczenia, podczas testowania program z nich nie korzysta, aby nie zafałszować wyników.

# Tworzenie sieci neuronowej oraz trenowanie

Sieć neuronowa jest tworzona na bazie zbioru treningowego. Wpierw trzeba podać bazę plików z tego zbioru (jako argumenty przyjąć ścieżki do folderu z adnotacjami .xml i folderu ze zdjęciami .png).

Następna linia definiuje model sieci, tj. słowa kluczowe, jakie będą wyszukiwane. Po tak zaimplementowanym modelu można przeprowadzić trenowanie zbioru oraz określić jego parametry dodatkowe. Obowiązkowo do podania jest argument dataset, natomiast liczba epochs domyślnie wynosi również 10, argument został uwzględniony celem szybszej jej ewentualnej zmiany.

```
#implementacja modelu sieci neuronowej i podanie slow kluczowych
#dataset_train = core.Dataset(train_annotations_path, train_images_path)
#model = core.Model(['speedlimit', 'crosswalk', 'stop', 'trafficlight'])

#trenowanie zbioru
#model.fit(dataset_train, epochs=10)
```

# WAŻNE!!!

Proces uczenia takiego zbioru zależy od wydajności posiadanego sprzętu. Celem jego przyspieszenia można wykorzystać właśnie narzędzie Google Colaboratory, aby połączyć się ze sprzętem dedykowanym pod uczenie maszynowe. Jako jednostkę odpowiedzialną za uczenie przyjąć jednostkę GPU. Wyniki przeprowadzenia uczenia dla wykorzystanych zbiorów są widoczne w pliku road_signs_detection.ipynb.

Natomiast wytrenowana już sieć znajduje się pod poniższym linkiem:

https://drive.google.com/file/d/1ykAhQhat0MC-3O-Ogd2JFgGFZwBhRW0g/view?usp=sharing

Została uzyskana po procesie uczenia za pomocą następującej komendy:

```
#zachowanie modelu sieci
#model.save('model_weights.pth')
```

Po pobraniu modelu sieci należy ją umieścić w katalogu głównym projektu (WdSI_projekt).

Z tego powodu linie kodu obejmujące trening sieci są domyślnie ZAKOMENTOWANE. Chcąc przeprowadzić uczenie, należy zakomentować poniższy kod dotyczący wczytania modelu tej sieci, a odkomentować te dotyczące treningu (zawierające dataset_train, model oraz model.fit).

```
#wczytanie modelu sieci
model = core.Model.load(project_path + '/' + 'model_weights.pth', ['speedlimit', 'crosswalk', 'stop', 'trafficlight'])
```

Sieć ta została wytrenowana z czasem ok. pół godziny (10 epochs x 3 min), czas trwania widoczny jest w pliku road_signs_detection.ipynb.

# Testy poprawności implementacji - zbiór treningowy

Dla sprawdzenia poprawności stworzonej sieci należy przeprowadzić szybki test detekcji obiektów na zdjęciu. Do tego celu służy poniższy kod:

```
#wywolanie obrazu ze zbioru treningowego celem sprawdzenia mozliwosci oznaczenia znakow
image_train = utils.read_image(train_images_path + '/' + 'road101.png')
predictions_train = model.predict(image_train)

labels_train, boxes_train, scores_train = predictions_train

#filtracja wynikow
train_filter = np.where(scores_train > 0.9)
scores_train_filter = scores_train[train_filter]
boxes_train_filter = boxes_train[train_filter]
num_list_train = list(train_filter[0])
labels_train_filter = []

for i in num_list_train:
    labels_train_filter.append(labels_train[i])

#wyplotowanie obrazu, rozpoznanie typu, podanie wspolrzednych oraz prawdopodobienstwa
print(labels_train_filter)
print(boxes_train_filter)
print(scores_train_filter)
visualize.show_labeled_image(image_train, boxes_train_filter, labels_train_filter)
```

Numer obrazu można dowolnie zmieniać w zakresie 0-658, aby sprawdzić inne zdjęcia.

# Testy poprawności implementacji - zbiór testowy

Wytrenowana sieć charakteryzuje się bardzo wysoką pewnością w detekcji obiektów, natomiast zdarza się, że ze względu na krótki czas treningu zostaną sklasyfikowane obiekty False Negative. W tym celu został stworzony filtr wyników, który przyjmuje tylko pewności powyżej 90% (przetestowane również na obrazach spoza zbioru testowego - losowych obrazach wyjętych z Google Images). Kod zbioru testowego de facto różni się ścieżką dostępu do obrazów:

```
#wywolanie obrazu ze zbioru testowego celem sprawdzenia mozliwości oznaczenia znakow
image_test = utils.read_image(test_images_path + '/' + 'road861.png')
predictions_test = model.predict(image_test)

labels_test, boxes_test, scores_test = predictions_test

#filtracja wynikow
test_filter = np.where(scores_test > 0.9)
scores_test_filter = scores_test[test_filter]
boxes_test_filter = boxes_test[test_filter]
num_list_test = list(test_filter[0])
labels_test_filter = []

for j in num_list_test:
    labels_test_filter.append(labels_test[j])

#wyplotowanie obrazu, rozpoznanie typu, podanie wspolrzednych oraz prawdopodobienstwa
print(labels_test_filter)
print(boxes_test_filter)
print(scores_test_filter)
visualize.show_labeled_image(image_test, boxes_test_filter, labels_test_filter)
```

Numer obrazu można dowolnie zmieniać w zakresie 659-876, aby sprawdzić inne zdjęcia.

# Detekcja

Właściwa funkcja detekcji jest zbliżona kodem do powyższego, odnoszącego się do zbioru testowego, lecz przekonfigurowana zgodnie z wytycznymi dotyczącymi przykładowego wyjścia.

```
#detekcja znakow ograniczenia predkosci
def detect():
    #petla wczytujaca obrazy ze zbioru testowego
    for detection in test_images_files:
        image_detect = utils.read_image(test_images_path + '/' + detection)
        predictions_detect = model.predict(image_detect)

        labels_detect, boxes_detect, scores_detect = predictions_detect

        #filtracja wynikow
        detect_filter = np.where(scores_detect > 0.9)
        scores_detect_filter = scores_detect[detect_filter]
        boxes_detect_filter = boxes_detect[detect_filter]
        num_list_detect = list(detect_filter[0])
        labels_detect_filter = []

        for k in num_list_detect:
            labels_detect_filter.append(labels_detect[k])

        #podanie nazwy obrazu, ilosci wykrytych obiektow oraz ich wspolrzednych
        print(detection)
        print(len(labels_detect_filter))
        print(boxes_detect_filter)        
```

Zmianie uległy elementy widoczne na wyjściu. Teraz ukazują się kolejno: nazwa odczytanego pliku, ilość obiektów wykrytych na zdjęciu oraz tensor ze współrzędnymi prostokąta/prostokątów z wykrytym obiektem/obiektami.

Wywołanie funkcji za pomocą standardowego wejścia zostało opatrzone poniższym kodem:

```
x = input()
if x == "detect":
    detect()
```

# Podsumowanie

Aby mieć podgląd graficzny na cały proces wykonywania kodu, przesyłam ponownie link do zbadania pliku road_signs_detection.ipynb:

https://github.com/Jakub-Pawinski-PP/WdSI_projekt/blob/main/road_signs_detection.ipynb

W razie wątpliwości lub potrzeby objaśnienia jestem otwarty na kontakt.
