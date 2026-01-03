import cv2
import time

img = cv2.imread(r"C:\Users\marcin\Desktop\SAHI_FK\demo\demo_data\small-vehicles1.jpeg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

H, W = gray.shape
podzial = 8
slice_h = H // podzial
slice_w = W // podzial

canny_th1 = 300
canny_th2 = 400
edge_threshold = 300

output = img.copy()

total = 0
kept = 0

start = time.time()

for y in range(0, H, slice_h):
    for x in range(0, W, slice_w):

        y_end = min(y + slice_h, H)
        x_end = min(x + slice_w, W)

        patch = gray[y:y_end, x:x_end]

        if patch.size == 0:
            continue

        total += 1
        edges = cv2.Canny(patch, canny_th1, canny_th2)

        if cv2.countNonZero(edges) > edge_threshold:
            kept += 1
            cv2.rectangle(output, (x, y), (x_end, y_end), (0, 255, 0), 1)
            cv2.putText(
                output,
                str(kept),
                (x + 5, y + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1
            )

        else:
            output[y:y_end, x:x_end] = 0



end = time.time()

print("Wycinki:", total)
print("Pozostale po FK:", kept)
print("Odrzucone:", total - kept)
print("Czas FK:", round(end - start, 3), "s")

cv2.imwrite("fk_visualization.png", output)
