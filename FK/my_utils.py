from os import path, makedirs
from datetime import date
from typing import Dict, List
dzien = date.today().strftime(r"%d-%m-%Y")

# "slice" - oryginalna metoda Wojcieszka (Canny per wycinek)
# "integral" - integral image
fk_mode = "slice"
filtruj_puste_wycinki: bool = True
liczba_wycinkow: int = 0
pozostale_wyc: int = 0
czasy_eval = list()
nr_zdj: int = 0

podzial: int = 6
nakladanie: float = 0.4
canny_th1: int = 300
canny_th2: int = 400
edgeThreshold: int = 300

zdjecia = path.join("zdjecia")
imgExtension = ".jpg"

slownik_modeli = {
  "8n":  path.join("models", "yolov8n.pt"),
  "8s":  path.join("models", "yolov8s.pt"),
  "8m":  path.join("models", "yolov8m.pt"),
  "8l":  path.join("models", "yolov8l.pt"),
  "8x":  path.join("models", "yolov8x.pt"),
  "11n": path.join("models", "yolo11n.pt"),
  "11s": path.join("models", "yolo11s.pt"),
  "11m": path.join("models", "yolo11m.pt"),
  "11l": path.join("models", "yolo11l.pt"),
  "11x": path.join("models", "yolo11x.pt")
}

wyniki = path.join("wyniki", dzien) 
if not path.exists(wyniki):
    makedirs(wyniki)
