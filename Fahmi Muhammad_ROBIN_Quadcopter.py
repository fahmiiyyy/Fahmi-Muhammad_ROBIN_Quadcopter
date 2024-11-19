import cv2
import numpy as np

# 
def put_text_centered(image, text, font, scale, color):
    text_size = cv2.getTextSize(text, font, scale, 2)[0]
    text_x = (image.shape[1] - text_size[0]) // 2
    text_y = (image.shape[0] + text_size[1]) // 2
    cv2.putText(image, text, (text_x, text_y), font, scale, color, 2, cv2.LINE_AA)

# 
def overlay_image_func(background, overlay, x, y):
    h, w, _ = overlay.shape
    alpha_overlay = overlay[:, :, 3] / 255.0  
    alpha_background = 1.0 - alpha_overlay

    for c in range(0, 3):
        background[y:y+h, x:x+w, c] = (alpha_overlay * overlay[:, :, c] + alpha_background * background[y:y+h, x:x+w, c])

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

overlay_img_path = r'C:\Users\fahmi\OneDrive\Dokumen\phy\.vscode\wokwok.png'  # Ganti dengan path gambar PNG yang Anda mau
overlay_img = cv2.imread(overlay_img_path, cv2.IMREAD_UNCHANGED)  

while True:
    ret, frame = cap.read()
    if not ret:
        break

    
    frame1 = frame.copy()  
    put_text_centered(frame1, "Fahmi ganteng parah", cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))

    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

    
    for (x, y, w, h) in faces:
        overlay_img_resized = cv2.resize(overlay_img, (w, h))
        overlay_image_func(frame1, overlay_img_resized, x, y)

    cv2.imshow('Frame 1 - Default', frame1)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Deteksi warna orange
    lower_orange = np.array([5, 150, 150])
    upper_orange = np.array([15, 255, 255])
    mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
    cv2.imshow('Frame 2 - Deteksi Warna Orange', mask_orange)

    # Deteksi warna merah
    lower_red1 = np.array([0, 100, 100])  
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    red_detected = cv2.bitwise_and(frame, frame, mask=mask_red)
    cv2.imshow('Frame 3 - Deteksi Warna Merah', red_detected)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
