from eval_setup import ewaluacja
import my_utils
from ultralytics import YOLO

my_utils.filtruj_puste_wycinki = True
my_utils.podzial, my_utils.nakladanie = 8 , 0.25
my_utils.canny_th1, my_utils.canny_th2 = 700, 700
ewaluacja(r"C:\Users\marcin\Desktop\SAHI_FK\FK\models\soda_n_final.pt", r"SODA-D/Annotations/test_eval.json", r"SODA-D/test/images", 0.05, output_dir="output_canny.txt")

my_utils.filtruj_puste_wycinki = True
my_utils.podzial, my_utils.nakladanie = 8 , 0.25
my_utils.canny_th1, my_utils.canny_th2 = 800, 800
ewaluacja(r"C:\Users\marcin\Desktop\SAHI_FK\FK\models\soda_n_final.pt", r"SODA-D/Annotations/test_eval.json", r"SODA-D/test/images", 0.05, output_dir="output_canny.txt")
