


class SSDDetection:

    def __init__(self,x = 0,y = 0,h = 0,w = 0,keypoints = None,confidence = 0):
        self.x = x
        self.y = y
        self.h = h
        self.w = w
        self.keypoints = keypoints
        self.confidence = confidence

    def getX(self):
        return self.x
    def getY(self):
        return self.y
    def getX2(self):
        return self.x + self.w
    def getY2(self):
        return self.y + self.y
    def getH(self):
        return self.h
    def getw(self):
        return self.w
    def getKeypoints(self):
        return self.keypoints
    def setKeypoints(self,keypoints):
        self.keypoints = keypoints
    def getConfidence(self):
        return self.confidence
    def setConfidence(self,confidence):
        self.confidence = confidence
    def setX(self,x):
        self.x = x
    def setY(self,y):
        self.y = y
    def setH(self,h):
        self.h = h
    def setW(self,w):
        self.w = w