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

def viewImage(image, name_of_window):
    cv2.namedWindow(name_of_window, cv2.WINDOW_NORMAL)
    cv2.imshow(name_of_window, image)

def nothing(*arg):
    pass

cv2.namedWindow("Crop")  # создаем главное окно
cv2.namedWindow("Number image")  # создаем главное окно
cv2.namedWindow("settings")  # создаем окно настроек

cv2.createTrackbar('Gauss', 'settings', 2, 4, nothing)
cv2.createTrackbar('Canny', 'settings', 1, 10, nothing)

#url = 'rtsp://192.168.1.167:8553/PSIA/Streaming/channels/1?videoCodecType=MPEG4'
#url = 'rtsp://192.168.1.167:8557/PSIA/Streaming/channels/2?videoCodecType=H.264'
#url = 'rtsp://admin:pP@697469@192.168.1.102:554/Streaming/Channels/101'
#cap = cv2.VideoCapture(url)
filename = r'c:\img\num1.rec'
cap = cv2.VideoCapture(filename)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
n = 0
keyList = [ord('q'),ord('w'),ord('a'),ord('s'),ord('d'),ord('f'),ord('g')]
step = 1

hsv_min = np.array((2, 0, 10), np.uint8)
hsv_max = np.array((240, 10, 240), np.uint8)

while (cap.isOpened()):

    for j in range(step+1):
        ret, im = cap.read()
    frame = im[110:429, 720:1199]
    gs = cv2.getTrackbarPos('Gauss', 'settings')
    gs = gs * 2 + 1
    cn = cv2.getTrackbarPos('Canny', 'settings')
    cn = cn * 10

    #img = cv2.GaussianBlur(frame, (gs, gs), 1.5)
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #img = cv2.erode(frame, kernel, iterations=1)
    #img = cv2.dilate(frame, kernel, iterations=1)
    #edge = cv2.Canny(img, cn, cn)

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Add some extra padding around the image
    #image = cv2.copyMakeBorder(image, 20, 20, 20, 20, cv2.BORDER_REPLICATE)
    # threshold the image (convert it to pure black and white)
    thresh = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    # find the contours (continuous blobs of pixels) the image
    contours,hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Hack for compatibility with different OpenCV versions
    #contours = contours[0] if imutils.is_cv2() else contours[1]
    letter_image_regions = []
    # Now we can loop through each of the four contours and extract the letter inside of each one
    for contour in contours:
        # Get the rectangle that contains the contour
        (x, y, w, h) = cv2.boundingRect(contour)
        # Compare the width and height of the contour to detect letters that
        # are conjoined into one chunk
        if (w > 10)and(h > 10):
            letter_image_regions.append((x, y, w, h))
        #if w / h > 1.25:
        #    # This contour is too wide to be a single letter!
        #    # Split it in half into two letter regions!
        #    half_width = int(w / 2)
        #    letter_image_regions.append((x, y, half_width, h))
        #    letter_image_regions.append((x + half_width, y, half_width, h))
        #else:
        #    # This is a normal letter by itself
        #    letter_image_regions.append((x, y, w, h))

    # If we found more or less than 4 letters in the captcha, our letter extraction
    # didn't work correcly. Skip the image instead of saving bad training data!
    #if len(letter_image_regions) != 4:
    #    continue
    # Sort the detected letter images based on the x coordinate to make sure
    # we are processing them from left-to-right so we match the right image
    # with the right letter
    letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])

    # Create an output image and a list to hold our predicted letters
    output = cv2.merge([image] * 3)

    #predictions = []

    # loop over the lektters
    for letter_bounding_box in letter_image_regions:
        # Grab the coordinates of the letter in the image
        x, y, w, h = letter_bounding_box
        # Extract the letter from the original image with a 2-pixel margin around the edge
        #letter_image = image[y - 2:y + h + 2, x - 2:x + w + 2]
        # Re-size the letter image to 20x20 pixels to match training data
        #letter_image = resize_to_fit(letter_image, 20, 20)
        # Turn the single image into a 4d list of images to make Keras happy
        #letter_image = np.expand_dims(letter_image, axis=2)
        #letter_image = np.expand_dims(letter_image, axis=0)
        # Ask the neural network to make a prediction
        #prediction = model.predict(letter_image)

        # Convert the one-hot-encoded prediction back to a normal letter
        #letter = lb.inverse_transform(prediction)[0]
        #predictions.append(letter)

        # draw the prediction on the output image
        cv2.rectangle(output, (x - 2, y - 2), (x + w + 4, y + h + 4), (0, 255, 0), 1)
        #cv2.putText(output, letter, (x - 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    # Print the captcha's text
    #captcha_text = "".join(predictions)
    #print("CAPTCHA text is: {}".format(captcha_text))


    #Вывод изображений
    cv2.imshow('Crop',frame)
    cv2.imshow("Number image", output)
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
