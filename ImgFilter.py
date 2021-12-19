import cv2
import numpy as np

drawing = False # true if mouse is pressed
mode = True # if True, draw rectangle. Press 'm' to toggle to curve
ix,iy = -1,-1

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

def change_shift_hue(hsv):
    #Делим на компоненты:
    h, s, v = cv2.split(hsv)
    #Поменять цвета вспять:
    val_h = 180 - h
    #Сдвиг цветов:
    val_h = (h + 90) % 180
    #Сдвиг цветов с маской:
    h_mask = (h < 75) | (h > 128)
    val_h[h_mask] = (h[h_mask] + 90) % 180

    res_hsv = cv2.merge([val_h, s, v])
    res_img = cv2.cvtColor(res_hsv, cv2.COLOR_HSV2BGR)
    return res_img

def viewImage(image, name_of_window):
    cv2.namedWindow(name_of_window, cv2.WINDOW_NORMAL)
    cv2.imshow(name_of_window, image)

def nothing(*arg):
    pass

cv2.namedWindow("Image")  # создаем главное окно
cv2.namedWindow("FilterOut")  # создаем главное окно
cv2.namedWindow("Edge")  # создаем главное окно
cv2.namedWindow("gauss")  # создаем окно настроек
cv2.namedWindow("hsv")  # создаем окно настроек

cv2.createTrackbar('Gauss', 'gauss', 2, 4, nothing)
cv2.createTrackbar('Canny', 'gauss', 1, 10, nothing)
cv2.createTrackbar('Hmin', 'hsv', 0, 180, nothing)
cv2.createTrackbar('Smin', 'hsv', 0, 240, nothing)
cv2.createTrackbar('Vmin', 'hsv', 0, 240, nothing)
cv2.createTrackbar('Hmax', 'hsv', 0, 240, nothing)
cv2.createTrackbar('Smax', 'hsv', 0, 255, nothing)
cv2.createTrackbar('Vmax', 'hsv', 0, 255, nothing)

filename = r'c:\img\num1.rec'
cap = cv2.VideoCapture(filename)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
n = 0
keyList = [ord('q'),ord('w'),ord('a'),ord('s'),ord('d'),ord('f'),ord('g')]
step = 1

#hsv_min = np.array((2, 0, 10), np.uint8)
#hsv_max = np.array((240, 10, 240), np.uint8)

#dark_mask = (0, 0, 64)
#light_mask = (255, 10, 192)

while (cap.isOpened()):

    if step>0 :
        for j in range(step+1):
            ret, im = cap.read()
    frame = im[110:429, 720:1199]
 
    gs = cv2.getTrackbarPos('Gauss', 'gauss')
    gs = gs * 2 + 1
    cn = cv2.getTrackbarPos('Canny', 'gauss')
    cn = cn * 10
    h_min = cv2.getTrackbarPos('Hmin', 'hsv')
    s_min = cv2.getTrackbarPos('Smin', 'hsv')
    v_min = cv2.getTrackbarPos('Vmin', 'hsv')
    h_max = cv2.getTrackbarPos('Hmax', 'hsv')
    s_max = cv2.getTrackbarPos('Smax', 'hsv')
    v_max = cv2.getTrackbarPos('Vmax', 'hsv')
    #h_max = cv2.getTrackbarPos('Hue', 'settings')+10

    #frame = cv2.medianBlur(frame, 7)
    #img = cv2.GaussianBlur(frame, (gs, gs), 1.5)
    #hsi = cv2.cvtColor(frame, cv2.COLOR_BGR2HSI)
    #mask = cv2.inRange(hsi, light_mask, dark_mask)
    #img = cv2.bitwise_and(frame, frame, mask=mask)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h = hsv[:,:,0]
    #s = hsv[:,:,1]
    #v = hsv[:,:,2]

    hist,bins = np.histogram(h,bins = np.linspace(0, 180, 19))

    print(hist)
    step = bins[1]

    index = np.argmax(hist)
    indexmin = index
    indexmax = index+1
    if indexmin>0 and hist[index]/hist[index-1]<3:
        indexmin = index - 1
    if indexmax<19 and hist[index]/hist[index+1]<3:
        indexmax = index + 1

    #авто        
    #h_min = indexmin * step
    #h_max = (indexmax+1) * step

    low_range = np.array((h_min, s_min, v_min), np.uint8)
    upper_range = np.array((h_max, s_max, v_max), np.uint8)

    thresh = cv2.inRange(hsv, low_range, upper_range)
    #thresh = ~thresh
    img = cv2.bitwise_and(frame, frame, mask=thresh)



    #img = cv2.GaussianBlur(img, (gs, gs), 1.5)
    #hsi = cv2.cvtColor(frame, cv2.COLOR_BGR2HSI)
    #mask = cv2.inRange(hsi, light_mask, dark_mask)
    #img = cv2.bitwise_and(frame, frame, mask=mask)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #img = cv2.erode(frame, kernel, iterations=1)
    #img = cv2.dilate(frame, kernel, iterations=1)
    edge = cv2.Canny(gray, cn, cn)

    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Add some extra padding around the image
    #image = cv2.copyMakeBorder(image, 20, 20, 20, 20, cv2.BORDER_REPLICATE)
    # threshold the image (convert it to pure black and white)
    thresh = cv2.threshold(image, 180, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours,hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    letter_image_regions = []
    for contour in contours:
        # Get the rectangle that contains the contour
        (x, y, w, h) = cv2.boundingRect(contour)
        if (w > 10)and(h > 10):
            letter_image_regions.append((x, y, w, h))

    letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])
    # переводим из pf8 в pf24
    output = cv2.merge([image] * 3) 

    for letter_bounding_box in letter_image_regions:
        # Grab the coordinates of the letter in the image
        x, y, w, h = letter_bounding_box
        cv2.rectangle(output, (x - 2, y - 2), (x + w + 4, y + h + 4), (0, 255, 0), 1)
    #Вывод изображений
    cv2.imshow('Image',frame)
    cv2.imshow("FilterOut", output)
    cv2.imshow("Edge", edge)
    key = 0
    while key not in keyList:
        key = cv2.waitKey(1) & 0xFF

    if key == ord('a'):
        step = 0
    if key == ord('s'):
        step = 1
    if key == ord('d'):
        step = 10
    if key == ord('f'):
        step = 100

    if key == ord('w'):
        #cropped = gray[0:319, 550:889]
        #img = cv2.GaussianBlur(cropped, (9, 9), 1.5)
        #s = "c:\coupler\coupler( {n} ).jpg".format(n=n)
        #cv2.imwrite(s,img)
        #cv2.imshow("Cropped image", img)
        n = n + 1
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
