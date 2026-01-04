import FK.my_utils as my_utils
from sahi_setup import sahi_fun

my_utils.filtruj_puste_wycinki = True
my_utils.canny_th1 = 300
my_utils.canny_th2 = 400
my_utils.podzial = 6   
my_utils.nakladanie = 0.4
my_utils.zdjecia = r"C:\Users\marcin\Desktop\SAHI_FK\demo\demo_data"
my_utils.imgExtension = ".jpeg"
my_utils.edgeThreshold = 300

result = sahi_fun(
    nazwa="small-vehicles1",
    auto_rozmiar=True,
    podzial=6,
    nakladanie=0.4,
    model="yolov8n",
    zapisz=True,
    doslowna_sciezka=False,
    full_prediction=True
)

