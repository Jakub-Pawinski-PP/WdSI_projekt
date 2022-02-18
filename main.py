import os
from detecto import core, utils, visualize
import numpy as np

#sciezka calego projektu
project_path = os.getcwd()

#sciezka zbioru treningowego
train_path = os.path.join(project_path, 'train')
train_annotations_path = os.path.join(train_path, 'annotations')
train_images_path = os.path.join(train_path, 'images')
train_annotations_files = os.listdir(train_annotations_path)
train_images_files = os.listdir(train_images_path)

#sciezka zbioru testowego
test_path = os.path.join(project_path, 'test')
test_annotations_path = os.path.join(test_path, 'annotations')
test_images_path = os.path.join(test_path, 'images')
test_annotations_files = os.listdir(test_annotations_path)
test_images_files = os.listdir(test_images_path)

#implementacja modelu sieci neuronowej i podanie slow kluczowych
dataset_train = core.Dataset(train_annotations_path, train_images_path)
model = core.Model(['speedlimit', 'crosswalk', 'stop', 'trafficlight'])

#trenowanie zbioru
model.fit(dataset_train, epochs=10)

#zachowanie modelu sieci
model.save('model_weights.pth')

#wczytanie modelu sieci
#model = core.Model.load(project_path + '/' + 'model_weights.pth', ['speedlimit', 'crosswalk', 'stop', 'trafficlight'])

"""
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
"""

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
        number_of_detected = len(labels_detect_filter)
        print(detection)
        print(number_of_detected)
        print(boxes_detect_filter)

x = input()
if x == "detect":
    detect()