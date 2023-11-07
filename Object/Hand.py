class Hand:

    def __init__(self,keypoints = None,confidence = 0,handness="none"):
        self.keypoints = keypoints
        self.confidence = confidence
        self.handness = handness