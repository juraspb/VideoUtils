import cv2
import numpy as np

#def sift_thread():
#	sift = cv2.xfeatures2d.SIFT_create()
#	(kps, descs) = sift.detectAndCompute(gray, None)
#	cv2.drawKeypoints(gray, kps, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#	cv2.imshow('SIFT Algorithm', img)


#def surf_thread():
#	surf = cv2.xfeatures2d.SURF_create()
#	(kps2, descs2) = surf.detectAndCompute(gray, None)
#	cv2.drawKeypoints(gray, kps2, img2, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#	cv2.imshow('SURF Algorithm', img2)

#def fast_thread():
#	fast = cv2.FastFeatureDetector_create()
#	kps3 = fast.detect(gray, None)
#	cv2.drawKeypoints(gray, kps3, img3, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#	cv2.imshow('FAST Algorithm', img3)

drawing = False # true if mouse is pressed
mode = True # if True, draw rectangle. Press 'm' to toggle to curve
ix,iy = -1,-1

# mouse callback function
def draw_box(event,x,y,flags,param):
    global ix,iy,drawing,mode

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            if mode == True:
                cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
            else:
                cv2.circle(img,(x,y),5,(0,0,255),-1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if mode == True:
            cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
        else:
            cv2.circle(img,(x,y),5,(0,0,255),-1)

def distance(f1, f2):    
    x1, y1 = f1.pt
    x2, y2 = f2.pt
    return np.sqrt((x2 - x1)**2+ (y2 - y1)**2)

def filteringByDistance(kp, distE=0.5):
    size = len(kp)
    mask = np.arange(1,size+1).astype(np.bool8) # all True   
    for i, f1 in enumerate(kp):
        if not mask[i]:
            continue
        else: # True
            for j, f2 in enumerate(kp):
                if i == j:
                    continue
                if distance(f1, f2)<distE:
                    mask[j] = False
    np_kp = np.array(kp)
    return list(np_kp[mask])

def viewImage(image, name_of_window):
    cv2.namedWindow(name_of_window, cv2.WINDOW_NORMAL)
    cv2.imshow(name_of_window, image)

def nothing(*arg):
    pass

cv2.namedWindow("hik_vision")  # создаем главное окно
cv2.setMouseCallback('hik_vision',draw_box)

cv2.namedWindow("Cropped image")  # создаем главное окно
cv2.namedWindow("settings")  # создаем окно настроек

cv2.createTrackbar('Gauss', 'settings', 2, 4, nothing)
cv2.createTrackbar('Canny', 'settings', 1, 10, nothing)

#cv2.namedWindow("SURF Keypoints")  # создаем главное окно
#cv2.namedWindow("dst")  # создаем главное окно

#detector = cv2.BRISK_create()
#fastF = cv2.FastFeatureDetector_create(threshold=30)
#mserF = cv2.MSER_create(10)
#blobF = cv2.SimpleBlobDetector_create()
#goodF = cv2.GFTTDetector_create(maxCorners= 20,minDistance = 10)
#detector = cv2.xfeatures2d.FREAK_create()

#url = 'rtsp://192.168.1.167:8553/PSIA/Streaming/channels/1?videoCodecType=MPEG4'
#url = 'rtsp://192.168.1.167:8557/PSIA/Streaming/channels/2?videoCodecType=H.264'
url = 'rtsp://admin:pP@697469@192.168.1.102:554/Streaming/Channels/101'
cap = cv2.VideoCapture(url)
#filename = 'c:\img\cap4.rec'
#cap = cv2.VideoCapture(filename)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
xstart = 1440 / 2 - 150
xend = 1440 / 2 + 150
n = 0
keyList = [ord('q'),ord('w'),ord('a'),ord('s'),ord('d'),ord('f'),ord('g')]
step = 1
while (cap.isOpened()):

    if step == 0:
        ret, frame = cap.read()
    else:
        for j in range(step+1):
            ret, frame = cap.read()

    gs = cv2.getTrackbarPos('Gauss', 'settings')
    gs = gs * 2 + 1
    cn = cv2.getTrackbarPos('Canny', 'settings')
    cn = cn * 10

    img = cv2.GaussianBlur(frame, (gs, gs), 1.5)
    #img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #img = cv2.erode(frame, kernel, iterations=1)
    #img = cv2.dilate(frame, kernel, iterations=1)

    edge = cv2.Canny(img, cn, cn)


    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    #filtered_kp, des = detector.compute(gray, filtered_kp)
    #img_keypoints = cv2.drawKeypoints(gray, filtered_kp, None, color=(0,0,255)) 
    #img_keypoints = cv2.drawKeypoints(gray, filtered_kp, frame, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) 
    #cv2.imshow('SURF Keypoints', img_keypoints)

    cv2.rectangle(frame, (500, 140), (540, 180), (0, 0, 255), 2)
    cv2.rectangle(frame, (900, 140), (940, 180), (0, 0, 255), 2)
    #Вывод изображений
    cv2.imshow('hik_vision',frame)

    key = 0
    while key not in keyList:
        key = cv2.waitKey(1) & 0xFF

    if key == ord('a'):
        step = 1
    if key == ord('s'):
        step = 10
    if key == ord('d'):
        step = 100
    if key == ord('f'):
        step = 1000

    if key == ord('w'):
        cropped = gray[0:319, 550:889]
        img = cv2.GaussianBlur(cropped, (9, 9), 1.5)

        s = "c:\coupler\coupler( {n} ).jpg".format(n=n)
        cv2.imwrite(s,img)
        n = n + 1
        cv2.imshow("Cropped image", img)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
