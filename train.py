import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import cv2
from cvzone.HandTrackingModule import HandDetector
import random


detector = HandDetector(maxHands=1)
boat_pred = ['Rock','Paper','Scizzor']

cam = cv2.VideoCapture(0)
clsfr = RandomForestClassifier()
data_Set = pickle.load(open('dataset.pickle','rb'))
with open('dataset.pickle', 'rb') as f:
    data_Set = pickle.load(f)

fl = False
pred_ls = []

ds = np.asarray(data_Set['data'])
lab = np.asarray(data_Set['label'])
#print(ds.shape)
#print(lab.size)

#80%-20% train test split ratio 
xtrain,xtest,ytrain,ytest = train_test_split(ds,lab,test_size=0.2,shuffle=True,stratify=lab)

clsfr.fit(xtrain,ytrain)

#predicting
#To test the accuracy of model
#pr = clsfr.predict(xtest)
#accuracy = accuracy_score(ytest, pr)
#print(f"Accuracy: {accuracy}")


while True:
    #read from camera and use the output frane to detect hands
    frm_flg,frame = cam.read()
    if not frm_flg:
        print("unable to capture camera")
        break
    hands,frame = detector.findHands(frame)

    if hands:
        if not fl:
            pred_val = random.randint(a=0,b=2)
            fl = True
        hand = hands[0]
        lnd_mrk_ls = hand['lmList']
        ms = []
        for lm in lnd_mrk_ls:
            ms.append(int(lm[1]))
            ms.append(int(lm[2]))
        prediction = clsfr.predict(np.asarray(ms).reshape(1,-1))[0]
        pred_ls.append(prediction)
    else:
        fl = False
        pred_ls=[]
        cv2.putText(frame, "No hand detected", (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)

    if len(pred_ls) >=10:
        #This create a list that checks the most recurrent prediction
        true_pred = str(max(set(pred_ls),key=pred_ls.count))
        boat_predicion = boat_pred[pred_val]
        if true_pred == "Rock":
            if str(boat_predicion) == "Scizzor":
                status_color = (0, 255, 0)
                result  = "YOU WON!"
            if str(boat_predicion) == "Paper":
                status_color = (0, 0, 255)
                result  = "BOAT WON!"
            if str(boat_predicion) == "Rock":
                status_color = (0, 255, 255)
                result  = "TIE!"
        elif true_pred == "Paper":
            if str(boat_predicion) == "Scizzor":
                status_color = (0, 0, 255)
                result  = "BOAT WON!"
            if str(boat_predicion) == "Rock":
                status_color = (0, 255, 0)
                result  = "YOU WON!"
            if str(boat_predicion) == "Paper":
                status_color = (0, 255, 255)
                result  = "TIE!"
        else:
            if str(boat_predicion) == "Scizzor":
                status_color = (0, 255, 255)
                result  = "TIE!"
            if str(boat_predicion) == "Rock":
                status_color = (0, 0, 255)
                result  = "BOAT WON!"
            if str(boat_predicion) == "Paper":
                status_color = (0, 255, 0)
                result  = "YOU WON!"
        cv2.putText(frame, true_pred, (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.putText(frame, f"bot predicted {boat_predicion}", (50, 400), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
        cv2.putText(frame, result, (50, 450), cv2.FONT_HERSHEY_PLAIN, 3, status_color, 3)
    cv2.imshow("video",frame)
    #getting the key we are using 27 since it's ascii of esc
    key = cv2.waitKey(1)
    if key==27:
        break


cam.release()
cv2.destroyAllWindows