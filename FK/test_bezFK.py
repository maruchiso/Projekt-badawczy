import my_utils
from sahi_setup import sahi_fun

my_utils.filtruj_puste_wycinki = True

result = sahi_fun(
    nazwa=r"C:\Users\marcin\Desktop\SAHI_FK\demo\demo_data\small-vehicles1.jpeg",
    podzial=6,
    nakladanie=0.4,
    zapisz=False,
    model="yolov8n",
    full_prediction=False,
    doslowna_sciezka=True,
    auto_rozmiar=True
)

print("Wycinki:", my_utils.liczba_wycinkow)
print("Pozostale:", my_utils.pozostale_wyc)
