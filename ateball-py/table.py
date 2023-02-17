import logging

import utils

class Hole:
    def __init__(self, name, image, center, offset):
        self.name = name
        self.image = image

        self.center = center
        self.offsetCenter = [center[0]+offset[0], center[1]+offset[1]]
        self.rotatedCenter = (0, 0)

        self.corner = (0,0)
        self.leftMarkH = (0,0)
        self.rightMarkH = (0,0)
        self.innerBoundLeft = (0,0)
        self.innerBoundRight = (0,0)

        self.holeGapSlope = 0
        self.holeGapSlopeRise = 0
        self.holeGapSlopeRun = 0

        self.hittablePointsToHole = []
        self.hittablePointsToHole1 = []
        self.hittablePointsToHole2 = []
        self.hittablePointsToHole3 = []

        self.points = [self.hittablePointsToHole1, self.hittablePointsToHole2, self.hittablePointsToHole3]

        self.generateBounds()
        self.generateHittablePoints()

        self.logger = logging.getLogger("ateball.hole")

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)

    def generateBounds(self):
        if self.name == "trh":
            self.leftMarkH = self.center[0] - 15, self.center[1] - 10
            self.rightMarkH = self.center[0] + 10, self.center[1] + 15
            self.innerBoundLeft = self.leftMarkH[0]+5, self.leftMarkH[1] - 10
            self.innerBoundRight = self.rightMarkH[0]+10, self.rightMarkH[1]-5
            self.corner = self.innerBoundRight[0], self.innerBoundLeft[1]
        elif self.name == "tmh":
            self.leftMarkH = self.center[0] - 21, self.center[1] + 10
            self.rightMarkH = self.center[0] + 21, self.center[1] + 10
            self.innerBoundLeft = self.leftMarkH[0], self.leftMarkH[1] - 10
            self.innerBoundRight = self.rightMarkH[0], self.rightMarkH[1] - 10
            self.corner = self.center
        elif self.name == "tlh":
            self.leftMarkH = self.center[0] - 10, self.center[1] + 15
            self.rightMarkH = self.center[0] + 15, self.center[1] - 10
            self.innerBoundLeft = self.leftMarkH[0]-10, self.leftMarkH[1]-5
            self.innerBoundRight = self.rightMarkH[0]-5, self.rightMarkH[1] - 10
            self.corner = self.innerBoundLeft[0], self.innerBoundRight[1]
        elif self.name == "blh":
            self.leftMarkH = self.center[0] + 15, self.center[1] + 10
            self.rightMarkH = self.center[0] - 10, self.center[1] - 15
            self.innerBoundLeft = self.leftMarkH[0]-5, self.leftMarkH[1] + 10
            self.innerBoundRight = self.rightMarkH[0]-10, self.rightMarkH[1]+5
            self.corner = self.innerBoundRight[0], self.innerBoundLeft[1]
        elif self.name == "bmh":
            self.leftMarkH = self.center[0] + 21, self.center[1] - 10
            self.rightMarkH = self.center[0] - 21, self.center[1] - 10
            self.innerBoundLeft = self.leftMarkH[0], self.leftMarkH[1] + 10
            self.innerBoundRight = self.rightMarkH[0], self.rightMarkH[1] + 10
            self.corner = self.center
        else:
            self.leftMarkH = self.center[0] + 10, self.center[1] - 15
            self.rightMarkH = self.center[0] - 15, self.center[1] + 10
            self.innerBoundLeft = self.leftMarkH[0]+10, self.leftMarkH[1]+5
            self.innerBoundRight = self.rightMarkH[0]+5, self.rightMarkH[1] + 10
            self.corner = self.innerBoundLeft[0], self.innerBoundRight[1]
            
        self.holeGapSlopeRise, self.holeGapSlopeRun, self.holeGapSlope = utils.PointHelper.findRiseRunSlope(self.leftMarkH, self.rightMarkH)

    def generateHittablePoints(self):
        minX = min(self.innerBoundLeft[0], self.innerBoundRight[0])
        maxX = max(self.innerBoundLeft[0], self.innerBoundRight[0])

        minY = min(self.innerBoundLeft[1], self.innerBoundRight[1])
        maxY = max(self.innerBoundLeft[1], self.innerBoundRight[1])

        if self.holeGapSlope > 0:
            y = minY
            increment = 1
        elif self.holeGapSlope < 0:
            y = maxY
            increment = -1
        else:
            y = minY
            increment = 0

        
        for x in range(minX, maxX+1, 1):
            targetPoint = (x, y)
            self.hittablePointsToHole.append(targetPoint)

            y += increment

        self.hittablePointsToHole.sort(key=lambda x: (x[0], x[1]))
        middle = len(self.hittablePointsToHole)//2
        eighth = len(self.hittablePointsToHole)//8
        sixteenth = len(self.hittablePointsToHole)//16

        # print(f"{self.hittablePointsToHole}")
        # print(f"{middle} - {eighth} - {sixteenth}")

        self.hittablePointsToHole1 = [self.hittablePointsToHole[0], self.hittablePointsToHole[middle], self.hittablePointsToHole[-1]]
        self.hittablePointsToHole2 = [self.hittablePointsToHole[eighth], self.hittablePointsToHole[eighth*3], self.hittablePointsToHole[eighth*5], self.hittablePointsToHole[eighth*7]]
        self.hittablePointsToHole3 = [self.hittablePointsToHole[sixteenth*3], self.hittablePointsToHole[sixteenth*5], self.hittablePointsToHole[sixteenth*11], self.hittablePointsToHole[sixteenth*13]]

class Wall:
    def __init__(self, startingPoint, endingPoint):
        self.startingPoint = startingPoint
        self.endingPoint = endingPoint

    def __str__(self):
        return "Wall"

    def __repr__(self):
        return str(self)
