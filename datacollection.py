import cv2
from cvzone.HandTrackingModule import HandDetector
import pickle

cam = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
crpd_win_flg = False
offset = 10
crp_w = 200
crp_h = 200
r_crpd_hnd = None

ds = []
ls = []


rn = 0
pn = 0
sn = 0

def get_landmark():
    ms = []
    for lm in lnd_mrk_ls:
        ms.append(int(lm[1]))
        ms.append(int(lm[2]))
    ds.append(ms)

while True:
    #read from camera and use the output frane to detect hands
    frm_flg,frame = cam.read()
    if not frm_flg:
        print("unable to capture camera")
        break
    hands,frame = detector.findHands(frame)

    if hands:
        hand = hands[0]
        x,y,w,h = hand['bbox']

        lnd_mrk_ls = hand['lmList']
        
        #draw a rectangle around hand
        #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        #gettting center point of x,y axis
        x_cntr = x+w//2
        y_cntr = y+h//2

        #defining the boarder
        crp_x_s = max(0,x_cntr-crp_w//2) #x has minimum of 0 atlease boarder, new starting x is calculated here
        crp_x_e = min(frame.shape[1],x_cntr+crp_w//2) #shape[1] has the width of our frame so its to not go out of boarder
        crp_y_s = max(0,y_cntr-crp_h//2)
        crp_y_e = min(frame.shape[0],y_cntr+crp_h//2)

        crpd_hnd = frame[crp_y_s:crp_y_e,crp_x_s:crp_x_e]

        if crpd_hnd.size>0:
            r_crpd_hnd = cv2.resize(crpd_hnd,(crp_w,crp_h))
            cv2.imshow("Cropped",r_crpd_hnd)
            crpd_win_flg = True
            key = cv2.waitKey(1)
        elif crpd_hnd.size==0 and crpd_win_flg ==True:
            cv2.destroyWindow(winname="Cropped")
            crpd_win_flg = False

    cv2.imshow("video",frame)
    #getting the key we are using 27 since it's ascii of esc
    key = cv2.waitKey(100)
    if key==27:
        break
    elif key==ord("r"):
        get_landmark()
        rn+=1
        ls.append('Rock')
        print(f"Rock {rn}Added")
    elif key==ord("p"):
        get_landmark()
        pn+=1
        ls.append('Paper')
        print(f"Paper {pn}Added")
    elif key==ord("s"):
        get_landmark()
        sn+=1
        ls.append('Scizzor')
        print(f"Scizzor {sn}Added")

cam.release()
cv2.destroyAllWindows

if ds:
    print("dataset created")
    f_data_set = open('dataset.pickle','wb')
    pickle.dump({'data':ds,'label':ls},f_data_set)
    f_data_set.close()
else:
    print("ds empty")