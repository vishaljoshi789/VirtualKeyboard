import cv2
from handTrakingModule import handDetector

cap = cv2.VideoCapture(0)

cap.set(3, 1280)
cap.set(4, 720)

detector = handDetector(detectionCon=0.8)

smallAlpha = [['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p'], ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', ';'], ['z', 'x', 'c', 'v', 'b', 'n', 'm', ',', '.', '/']]
captalAlpha = [['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'], ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L', ';'], ['Z', 'X', 'C', 'V', 'B', 'N', 'M', ',', '.', '/']]

def draw(img, buttonList):
    for button in buttonList:
        x, y = button.pos
        w, h = button.size
        cv2.rectangle(img,button.pos, (x+w, y+h), (255,0,255), cv2.FILLED)
        cv2.putText(img, button.text, (x+20, y+65), cv2.FONT_HERSHEY_PLAIN, 4, (255,255,255), 4)
    return img

class button():
    def __init__(self, pos, text, size=[85,85]):
        self.pos = pos
        self.text = text
        self.size = size
        



buttonList = []
for i in range(len(smallAlpha)):
    for j, key in enumerate(smallAlpha[i]):
        buttonList.append(button([100*j + 50, 100 *i +50], key))


while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmlist = detector.findPosition(img)
    img = draw(img, buttonList)

    if lmlist:
        for button in buttonList:
            x, y = button.pos
            w, h = button.size

            if x<lmlist[8][1]<x+w and y<lmlist[8][2]<y+h:
                cv2.rectangle(img,button.pos, (x+w, y+h), (0,255,8), cv2.FILLED)
                cv2.putText(img, button.text, (x+20, y+65), cv2.FONT_HERSHEY_PLAIN, 4, (255,255,255), 4)
                l, _ = detector.findDistance([lmlist[8][1], lmlist[8][2]], [lmlist[12][1],lmlist[12][2]], img)
                if l<40:
                    cv2.rectangle(img,button.pos, (x+w, y+h), (0,255,255), cv2.FILLED)
                    cv2.putText(img, button.text, (x+20, y+65), cv2.FONT_HERSHEY_PLAIN, 4, (0,0,0), 4)

    cv2.imshow('Image', img)
    cv2.waitKey(1)