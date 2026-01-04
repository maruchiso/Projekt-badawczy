from sahi.predict import get_sliced_prediction
import FK.my_utils as my_utils
from PIL import Image
import time

# wykonaj inferencje i zapisz wynik
def sahi_fun_eval(
         model,
         jpg_path,
         nakladanie = 0.2, 
         NMS = 'NMS', 
         czulosc_IOU = 0.05,
         podzial = 4, 
         full_prediction = True):

    # policz rozmiar wycinka
    szerokosc, wysokosc = policz_rozmiar_wycinkow(jpg_path, podzial)

     
    start = time.time()

    # wyniki inferencji
    result = get_sliced_prediction(
        jpg_path,
        model,
        slice_height = wysokosc,
        slice_width  = szerokosc,
        overlap_height_ratio = nakladanie,
        overlap_width_ratio = nakladanie,
        perform_standard_pred = full_prediction,
        postprocess_type = NMS, # -> "GREEDYNMM", 'NMM', 'NMS', LSNMS??
        postprocess_match_metric = "IOU",
        postprocess_match_threshold = czulosc_IOU,
        postprocess_class_agnostic = False,
        verbose = 0, ###################### zmienione z 2
        # merge_buffer_length = None,
        auto_slice_resolution = False 
        )

    # mierzymy czas
    czas: float = time.time() - start

    my_utils.czasy_eval.append(czas)

    return result


def policz_rozmiar_wycinkow(sciezka_zdjecia: str, podzial:int) -> tuple[int, int]:
    zdjecie = Image.open(sciezka_zdjecia)
    szerokosc, wysokosc = zdjecie.size
    szerokosc = szerokosc//podzial
    wysokosc = wysokosc//podzial
    return szerokosc, wysokosc
