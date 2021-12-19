import cv2
import numpy as np

def nothing(*arg):
    pass

cv2.namedWindow("hik_vision")  # создаем главное окно

# cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FPS, 24) # Частота кадров
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640) # Ширина кадров в видеопотоке.
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # Высота кадров в видеопотоке.

#url = 'rtsp://192.168.1.167:8553/PSIA/Streaming/channels/1?videoCodecType=MPEG4'
#url = 'rtsp://192.168.1.167:8557/PSIA/Streaming/channels/2?videoCodecType=H.264'
url = 'rtsp://admin:pP@697469@192.168.1.102:554/Streaming/Channels/101'
#url = 'rtsp://admin:pP@697469@192.168.1.102:554'
cap = cv2.VideoCapture(url)
#filename = r'c:/img/num1.rec'
#filename = 'cap.avi'
#cap = cv2.VideoCapture(filename)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

codec = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(r'img/captured.avi',codec, 25.0, (width,height))
#codec = cv2.VideoWriter_fourcc(*'MJPG')
#out = cv2.VideoWriter('img/captured.mp4',codec, 25.0, (width,height))

while (cap.isOpened()):

    ret, frame = cap.read()
    #crop 320x480
    #crop = frame[110:430, 720:1200]

    #Вывод изображений
    #cv2.rectangle(frame, (740,120), (1170,420), (0, 0, 255), 5)
    cv2.imshow('hik_vision',frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
out.release()
cap.release()
cv2.destroyAllWindows()
