from sahi import AutoDetectionModel 
from sahi.predict import get_sliced_prediction
from sahi.prediction import PredictionResult 
import FK.my_utils as my_utils
from my_utils import path
from PIL import Image
import time

# wykonaj inferencje i zapisz wynik
def sahi_fun(nazwa = "Neapol", 
         szerokosc = 512, 
         nakladanie = 0.2, 
         NMS = 'NMS', 
         czulosc = 0.05, 
         czulosc_modelu = 0.2,
         auto_rozmiar = False, 
         podzial = 4, 
         etykiety = False, 
         komentarz = "", 
         model = "yolov11n", 
         zapisz = True,
         doslowna_sciezka = False,
         doslowny_model = False,
         full_prediction = True):

    model_nazwa: str = model
    if model in my_utils.slownik_modeli: model = my_utils.slownik_modeli[model]
    elif(doslowny_model): 
        model = model
        model_nazwa: str = "model4"
    else:  model = path.join("models", f"{model}")

    # stworz obiekt klasy yolo
    model_sahi = stworz_model(model, czulosc_modelu)
    
    # sciezka do aktualnego zdjecia
    if doslowna_sciezka: sciezka_zdjecia = nazwa
    else: sciezka_zdjecia = path.join(my_utils.zdjecia, str(nazwa) + my_utils.imgExtension)

    # policz rozmiar wycinka
    if(auto_rozmiar): szerokosc, wysokosc = policz_rozmiar_wycinkow(sciezka_zdjecia, podzial)

    # wlasciwa analiza
    result, czas, liczba_obiektow = szukaj_obiektow(sciezka_zdjecia, wysokosc, szerokosc, model_sahi, nakladanie, NMS, czulosc, full_prediction)

    # zapisz zdjecie
    if doslowna_sciezka:
        my_utils.nr_zdj += 1
        nazwa = my_utils.nr_zdj
    nazwa_pliku = nowa_nazwa(nazwa, liczba_obiektow, model_nazwa, NMS, czas, komentarz)
    print(nazwa_pliku)

    if(zapisz): zapisz_zdjecie(result, nazwa_pliku, etykiety)

    return result




def stworz_model(model: str, czulosc_modelu: float):
        model_sahi = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=model,
        #model_path="yolo11x.pt",
        confidence_threshold=czulosc_modelu,
        device="cuda:0", # lub 'CPU'
        )
        return model_sahi

# nazwa pliku jpg, który będzie zapisany
def nowa_nazwa(nazwa, liczba_obiektow: int, model: str, NMS: str, czas: float, komentarz: str) -> str:
    return (f"{nazwa}_sahi_s{my_utils.liczba_wycinkow}_o{liczba_obiektow}_{model}_{NMS}_{czas:.2f}{komentarz}")

def zapisz_zdjecie(result, nazwa_pliku: str, etykiety: bool) -> None:
     # ustawienia estetyki
    result.export_visuals(my_utils.wyniki, hide_labels = True, hide_conf = True, file_name = nazwa_pliku, rect_th = 4)
    if(etykiety):
        result.export_visuals(my_utils.wyniki, hide_labels = False, hide_conf = False, file_name = nazwa_pliku + "_etykiety", rect_th = 1, text_size=0.3)
    # print("")

def policz_rozmiar_wycinkow(sciezka_zdjecia: str, podzial:int) -> tuple[int, int]:
    zdjecie = Image.open(sciezka_zdjecia)
    szerokosc, wysokosc = zdjecie.size
    szerokosc = int(szerokosc/podzial)
    wysokosc = int(wysokosc/podzial) 
    return szerokosc, wysokosc

def szukaj_obiektow(sciezka_zdjecia: str, 
                    wysokosc: int, 
                    szerokosc:int, 
                    model_sahi, 
                    nakladanie: float, 
                    NMS: str, 
                    czulosc: float,
                    full_prediction: bool) -> tuple[PredictionResult, float, int]:
     
    start = time.time()

    # wyniki inferencji
    result = get_sliced_prediction(
        sciezka_zdjecia,
        model_sahi,
        slice_height = wysokosc,
        slice_width = szerokosc,
        overlap_height_ratio = nakladanie,
        overlap_width_ratio = nakladanie,
        perform_standard_pred = full_prediction,
        postprocess_type = NMS, # -> "GREEDYNMM", 'NMM', 'NMS', LSNMS??
        postprocess_match_metric = "IOU",
        postprocess_match_threshold = czulosc,
        postprocess_class_agnostic = False,
        verbose = 2, ###################### zmienione z 2
        # merge_buffer_length = None,
        auto_slice_resolution = False 
        )

    # mierzymy czas
    end: float = time.time()
    czas: float = end - start

    #my_utils.czasy_eval.append(czas)

    liczba_obiektow:int = len(result.object_prediction_list)

    print("czas: " + "%.2f" % czas)
    print("Liczba obiektów: " + str(liczba_obiektow))

    return result, czas, liczba_obiektow
