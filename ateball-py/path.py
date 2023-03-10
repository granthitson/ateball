import logging

import cv2
import numpy as np

import utils
from constants import constants

logger = logging.getLogger("ateball.path")

class Path:
    def __init__(self, cue, ball, hole, targetPoint, imgPath):
        self.imgPath = imgPath

        self.hitTypeToTarget = "single"
        self.hitTypeToHole = "single"
        
        self.cueball = cue
        self.ball = ball
        self.hole = hole

        self.difficulty = 0 #out of 10
        self.benefit = 0 #out of 10

        if self.hitTypeToTarget == "multi":
            self.updateDifficulty(1)

        self.targetPoint = targetPoint

        self.hitPoint = (0, 0)
        self.hitPointOffset = (0,0)
        self.hitPointStart = (0, 0)
        self.hitPointEnd = (0, 0)
        self.angle = 0

        self.closeToHole = False

        # slope to hole
        self.distanceToHole = 0
        self.slopeToHoleRise = 0
        self.slopeToHoleRun = 0
        self.slopeToHole = 0
        self.perpSlopeH = 0

        # points on ball to hole
        self.leftMarkB1 = (0, 0)
        self.rightMarkB1 = (0, 0)

        # points on ball to cue
        self.leftMarkB2 = (0, 0)
        self.rightMarkB2 = (0, 0)

        # slope to ball
        self.distanceToHP = 0
        self.slopeToHPRise = 0
        self.slopeToHPRun = 0
        self.slopeToHP = 0
        self.perpSlopeHP = 0

        # points on cue to ball
        self.leftMarkC = (0, 0)
        self.rightMarkC = (0, 0)

        self.blockingCueToBall = []
        self.blockingBallToHole = []
        self.alternateTargets = []

        self.setupParamsFromBallToHole()
        self.setupParamsFromCueToBall()

    def __str__(self):
        return f"shot on {self.ball.name} to {self.hole.name} - benefit: {self.benefit} - difficulty: {self.difficulty}"

    def __repr__(self):
        return str(self)

    def updateDifficulty(self, add=1):
        self.difficulty += add

    def updateBenefit(self, add=1):
        self.benefit += add

    def setupParamsFromBallToHole(self):
        self.distanceToHole = utils.measureDistance(self.ball.center, self.targetPoint)
        self.slopeToHoleRise, self.slopeToHoleRun, self.slopeToHole = utils.findRiseRunSlope(self.targetPoint, self.ball.center)

        if self.slopeToHole != 0:
            self.perpSlopeH = (-1 / self.slopeToHole)
        else:
            self.perpSlopeH = 0

        if self.slopeToHoleRise < 0:
            self.leftMarkB1, self.rightMarkB1 = utils.findPointsOnEitherSideOf(self.ball.center, 10, -self.slopeToHoleRun, self.slopeToHoleRise)
        else:
            self.leftMarkB1, self.rightMarkB1 = utils.findPointsOnEitherSideOf(self.ball.center, 10, -self.slopeToHoleRun, self.slopeToHoleRise, True)

        if self.slopeToHoleRise < 0:
            if self.slopeToHoleRun > 0:
                x, y = utils.findPointsAlongSlope(self.ball.center, 20, self.slopeToHole, True)
            else:
                x, y = utils.findPointsAlongSlope(self.ball.center, 20, self.slopeToHole)
        else:
            if self.slopeToHoleRun < 0:
                x, y = utils.findPointsAlongSlope(self.ball.center, 20, self.slopeToHole)
            else:
                x, y = utils.findPointsAlongSlope(self.ball.center, 20, self.slopeToHole, True)

        self.hitPoint = utils.roundTuple((x, y))
        self.hitPointOffset = (x+constants.offset[0]), (y+constants.offset[1])

    def setupParamsFromCueToBall(self):
        self.distanceToHP = utils.measureDistance(self.hitPoint, self.cueball.center)
        self.slopeToHPRise, self.slopeToHPRun, self.slopeToHP = utils.findRiseRunSlope(self.cueball.center, self.hitPoint)

        if self.slopeToHP != 0:
            self.perpSlopeHP = (-1 / self.slopeToHP)
        else:
            self.perpSlopeHP = 0

        if self.slopeToHP < 0:
            self.leftMarkC, self.rightMarkC = utils.findPointsOnEitherSideOf(self.cueball.center, 10, -self.slopeToHPRun, self.slopeToHPRise)
        elif self.slopeToHP > 0:
            self.leftMarkC, self.rightMarkC = utils.findPointsOnEitherSideOf(self.cueball.center, 10, -self.slopeToHPRun, self.slopeToHPRise, True)
        else: #unsure
            self.leftMarkC, self.rightMarkC = utils.findPointsOnEitherSideOf(self.cueball.center, 10, -self.slopeToHPRun, self.slopeToHPRise, True)

        if self.slopeToHole < 0:
            if self.slopeToHP < 0:
                self.leftMarkB2, self.rightMarkB2 = utils.findPointsOnEitherSideOf(self.hitPoint, 10, -self.slopeToHPRun, self.slopeToHPRise)
            else: 
                self.leftMarkB2, self.rightMarkB2 = utils.findPointsOnEitherSideOf(self.hitPoint, 10, -self.slopeToHPRun, self.slopeToHPRise, True)
        elif self.slopeToHole > 0:
            if self.slopeToHP < 0:
                self.leftMarkB2, self.rightMarkB2 = utils.findPointsOnEitherSideOf(self.hitPoint, 10, -self.slopeToHPRun, self.slopeToHPRise)
            else: 
                self.leftMarkB2, self.rightMarkB2 = utils.findPointsOnEitherSideOf(self.hitPoint, 10, -self.slopeToHPRun, self.slopeToHPRise, True)

    def isHittable(self, region, region1):
        ''' Is the point you hit the ball within a certain region?'''

        minX, maxX = region[0], region[0] + region[2]
        minY, maxY = region[1], region[1] + region[3]

        self.setupParamsFromBallToHole()

        if maxX >= self.hitPointOffset[0] >= minX and minY <= self.hitPointOffset[1] <= maxY:
            self.updateDifficulty(1)
            return True
        else:
            minX, maxX = region1[0], region1[0] + region1[2]
            minY, maxY = region1[1], region1[1] + region1[3]

            distanceToHole = utils.measureDistance(self.ball.center, self.hole.center)
            if distanceToHole < 20 and (maxX >= self.hitPointOffset[0] >= minX and minY <= self.hitPointOffset[1] <= maxY):
                self.updateBenefit(5)
                return True

        self.updateDifficulty(999)

        return False  

    def isViableTarget(self):
        '''Is the hole point you are targeting a viable option? Will it hit a wall?'''
        clean = cv2.imread(self.imgPath + "pooltable.png", 1)
        cv2.line(clean, utils.tupleToInt(self.ball.center), utils.tupleToInt(self.targetPoint),(255, 255, 255), 1)

        angle = utils.findAngle(self.targetPoint, self.ball.center, (self.ball.center[0], self.targetPoint[1]))
        rise, run, slope = utils.findRiseRunSlope(self.targetPoint, self.ball.center)
        if slope != 0:
            if slope < 0:
                angle = -angle
        else:
            if rise == 0:
                    angle = -90

        if "t" in self.hole.name:
            if "r" in self.hole.name:
                clampL = utils.clamp(self.targetPoint[1], self.hole.corner[1], self.hole.leftMarkH[1])
                clampR = utils.clamp(self.targetPoint[0], self.hole.rightMarkH[0], self.hole.corner[0])
                
                minAngleL = utils.findAngle((self.targetPoint[0]-100, clampL), (self.targetPoint[0], clampL), self.hole.leftMarkH)
                minAngleR = utils.findAngle((clampR, self.targetPoint[1]+100), (clampR, self.targetPoint[1]), self.hole.rightMarkH)

                angleL = utils.findAngle((self.hole.leftMarkH[0]-100, self.hole.leftMarkH[1]), self.hole.leftMarkH, self.leftMarkB1)
                angleR = utils.findAngle((self.hole.rightMarkH[0], self.hole.rightMarkH[1]+100), self.hole.rightMarkH, self.rightMarkB1)

                cv2.line(clean, utils.tupleToInt((self.targetPoint[0]-100, clampL)), utils.tupleToInt((self.targetPoint[0], clampL)),(255, 0, 0), 1)
                cv2.line(clean, utils.tupleToInt((self.targetPoint[0], clampL)), utils.tupleToInt(self.hole.leftMarkH),(255, 0, 0), 1)

                cv2.line(clean, utils.tupleToInt((clampR, self.targetPoint[1]+100)), utils.tupleToInt((clampR, self.targetPoint[1])),(255, 255, 0), 1)
                cv2.line(clean, utils.tupleToInt((clampR, self.targetPoint[1])), utils.tupleToInt(self.hole.rightMarkH),(255, 255, 0), 1)
            elif "m" in self.hole.name:
                clampL = utils.clamp(self.targetPoint[1], self.hole.corner[1], self.hole.leftMarkH[1])
                clampR = utils.clamp(self.targetPoint[1], self.hole.corner[1], self.hole.rightMarkH[1])
                
                minAngleL = utils.findAngle((self.targetPoint[0]-100, clampL), (self.targetPoint[0], clampL), self.hole.leftMarkH)
                minAngleR = utils.findAngle((self.targetPoint[0]+100, clampR), (self.targetPoint[0], clampR), self.hole.rightMarkH)

                angleL = utils.findAngle((self.hole.leftMarkH[0]-100, self.hole.leftMarkH[1]), self.hole.leftMarkH, self.leftMarkB1)
                angleR = utils.findAngle((self.hole.rightMarkH[0]+100, self.hole.rightMarkH[1]), self.hole.rightMarkH, self.rightMarkB1)

                cv2.line(clean, utils.tupleToInt((self.targetPoint[0]-100, clampL)), utils.tupleToInt((self.targetPoint[0], clampL)),(255, 0, 0), 1)
                cv2.line(clean, utils.tupleToInt((self.targetPoint[0], clampL)), utils.tupleToInt(self.hole.leftMarkH),(255, 0, 0), 1)

                cv2.line(clean, utils.tupleToInt((self.targetPoint[0]+100, clampR)), utils.tupleToInt((self.targetPoint[0], clampR)),(255, 255, 0), 1)
                cv2.line(clean, utils.tupleToInt((self.targetPoint[0], clampR)), utils.tupleToInt(self.hole.rightMarkH),(255, 255, 0), 1)
            else:
                clampL = utils.clamp(self.targetPoint[0], self.hole.corner[0], self.hole.leftMarkH[0])
                clampR = utils.clamp(self.targetPoint[1], self.hole.corner[1], self.hole.rightMarkH[1])
                
                minAngleL = utils.findAngle((clampL, self.targetPoint[1]+100), (clampL, self.targetPoint[1]), self.hole.leftMarkH)
                minAngleR = utils.findAngle((self.targetPoint[0]+100, clampR), (self.targetPoint[0], clampR), self.hole.rightMarkH)
                
                angleL = utils.findAngle((self.hole.leftMarkH[0], self.hole.leftMarkH[1]+100), self.hole.leftMarkH, self.leftMarkB1)
                angleR = utils.findAngle((self.hole.rightMarkH[0]+100, self.hole.rightMarkH[1]), self.hole.rightMarkH, self.rightMarkB1)
                
                cv2.line(clean, utils.tupleToInt((clampL, self.targetPoint[1]+100)), utils.tupleToInt((clampL, self.targetPoint[1])),(255, 0, 0), 1)
                cv2.line(clean, utils.tupleToInt((clampL, self.targetPoint[1])), utils.tupleToInt(self.hole.leftMarkH),(255, 0, 0), 1)

                cv2.line(clean, utils.tupleToInt((self.targetPoint[0]+100, clampR)), utils.tupleToInt((self.targetPoint[0], clampR)),(255, 255, 0), 1)
                cv2.line(clean, utils.tupleToInt((self.targetPoint[0], clampR)), utils.tupleToInt(self.hole.rightMarkH),(255, 255, 0), 1)
        else:
            if "r" in self.hole.name:
                clampL = utils.clamp(self.targetPoint[0], self.hole.leftMarkH[0], self.hole.corner[0])
                clampR = utils.clamp(self.targetPoint[1], self.hole.rightMarkH[1], self.hole.corner[1])
                
                minAngleL = utils.findAngle((clampL, self.targetPoint[1]-100), (clampL, self.targetPoint[1]), self.hole.leftMarkH)
                minAngleR = utils.findAngle((self.targetPoint[0]-100, clampR), (self.targetPoint[0], clampR), self.hole.rightMarkH)
                
                angleL = utils.findAngle((self.hole.leftMarkH[0], self.hole.leftMarkH[1]-100), self.hole.leftMarkH, self.leftMarkB1)
                angleR = utils.findAngle((self.hole.rightMarkH[0]-100, self.hole.rightMarkH[1]), self.hole.rightMarkH, self.rightMarkB1)

                cv2.line(clean, utils.tupleToInt((clampL, self.targetPoint[1]-100)), utils.tupleToInt((clampL, self.targetPoint[1])),(255, 0, 0), 1)
                cv2.line(clean, utils.tupleToInt((clampL, self.targetPoint[1])), utils.tupleToInt(self.hole.leftMarkH),(255, 0, 0), 1)

                cv2.line(clean, utils.tupleToInt((self.targetPoint[0]-100, clampR)), utils.tupleToInt((self.targetPoint[0], clampR)),(255, 255, 0), 1)
                cv2.line(clean, utils.tupleToInt((self.targetPoint[0], clampR)), utils.tupleToInt(self.hole.rightMarkH),(255, 255, 0), 1)
            elif "m" in self.hole.name:
                clampL = utils.clamp(self.targetPoint[1], self.hole.leftMarkH[1], self.hole.corner[1])
                clampR = utils.clamp(self.targetPoint[1], self.hole.corner[1], self.hole.rightMarkH[1])
                
                minAngleL = utils.findAngle((self.targetPoint[0]+100, clampL), (self.targetPoint[0], clampL), self.hole.leftMarkH)
                minAngleR = utils.findAngle((self.targetPoint[0]-100, clampR), (self.targetPoint[0], clampR), self.hole.rightMarkH)

                angleL = utils.findAngle((self.hole.leftMarkH[0]+100, self.hole.leftMarkH[1]), self.hole.leftMarkH, self.leftMarkB1)
                angleR = utils.findAngle((self.hole.rightMarkH[0]-100, self.hole.rightMarkH[1]), self.hole.rightMarkH, self.rightMarkB1)

                cv2.line(clean, utils.tupleToInt((self.targetPoint[0]+100, clampL)), utils.tupleToInt((self.targetPoint[0], clampL)),(255, 0, 0), 1)
                cv2.line(clean, utils.tupleToInt((self.targetPoint[0], clampL)), utils.tupleToInt(self.hole.leftMarkH),(255, 0, 0), 1)

                cv2.line(clean, utils.tupleToInt((self.targetPoint[0]-100, clampR)), utils.tupleToInt((self.targetPoint[0], clampR)),(255, 255, 0), 1)
                cv2.line(clean, utils.tupleToInt((self.targetPoint[0], clampR)), utils.tupleToInt(self.hole.rightMarkH),(255, 255, 0), 1)
            else:
                clampL = utils.clamp(self.targetPoint[1], self.hole.leftMarkH[1], self.hole.corner[1])
                clampR = utils.clamp(self.targetPoint[0], self.hole.corner[0], self.hole.rightMarkH[0])
                
                minAngleL = utils.findAngle((self.targetPoint[0]+100, clampL), (self.targetPoint[0], clampL), self.hole.leftMarkH)
                minAngleR = utils.findAngle((clampR, self.targetPoint[1]-100), (clampR, self.targetPoint[1]), self.hole.rightMarkH)

                angleL = utils.findAngle((self.hole.leftMarkH[0]+100, self.hole.leftMarkH[1]), self.hole.leftMarkH, self.leftMarkB1)
                angleR = utils.findAngle((self.hole.rightMarkH[0], self.hole.rightMarkH[1]-100), self.hole.rightMarkH, self.rightMarkB1)

                cv2.line(clean, utils.tupleToInt((self.targetPoint[0]+100, clampL)), utils.tupleToInt((self.targetPoint[0], clampL)),(255, 0, 0), 1)
                cv2.line(clean, utils.tupleToInt((self.targetPoint[0], clampL)), utils.tupleToInt(self.hole.leftMarkH),(255, 0, 0), 1)

                cv2.line(clean, utils.tupleToInt((clampR, self.targetPoint[1]-100)), utils.tupleToInt((clampR, self.targetPoint[1])),(255, 255, 0), 1)
                cv2.line(clean, utils.tupleToInt((clampR, self.targetPoint[1])), utils.tupleToInt(self.hole.rightMarkH),(255, 255, 0), 1)

        rotatedLeftMarkH = utils.rotateAround(self.ball.center, self.hole.leftMarkH, angle)
        rotatedRightMarkH = utils.rotateAround(self.ball.center, self.hole.rightMarkH, angle)

        distance1 = utils.measureDistance(rotatedLeftMarkH, (self.ball.center[0], rotatedLeftMarkH[1]))
        distance2 = utils.measureDistance(rotatedRightMarkH, (self.ball.center[0], rotatedRightMarkH[1]))

        if "m" in self.hole.name:
            minDistanceL = (utils.clamp((180 / angleL) * 10, 0, 10))
            minDistanceR = (utils.clamp((180 / angleR) * 10, 0, 10))
        else:
            minDistanceL = (utils.clamp((55 / angleL) * 10, 0, 10))
            minDistanceR = (utils.clamp((55 / angleR) * 10, 0, 10))

        # print(self.hitPoint)
        # print(angleL, minAngleL, distance1, minDistanceL, (self.targetPoint[0]-100, utils.clamp(self.targetPoint[1], self.hole.corner[1], self.hole.leftMarkH[1])))
        # print(angleR, minAngleR, distance2, minDistanceR, (utils.clamp(self.targetPoint[0], self.hole.rightMarkH[0], self.hole.corner[0]), self.targetPoint[1]+100))

        cv2.imshow('test', clean) 
        cv2.waitKey(0)
        cv2.destroyAllWindows() 

        if (angleL > minAngleL and distance1 > minDistanceL) and (angleR > minAngleR and distance2 > minDistanceR):
            self.updateBenefit(1)
            return True

        self.updateDifficulty(5)
        return False

    def checkPathofBall(self, unpocketed):
        self.blockingBallToHole = self.findBallsBlockingHole(unpocketed)
        if len(self.blockingBallToHole) > 0:
            self.updateDifficulty(len(self.blockingBallToHole))

            self.alternateTargets = []
            for ball in self.blockingBallToHole:
                if ball.target is True and ball in unpocketed:
                    self.alternateTargets.append(ball)

            # print(self.alternateTargets)
            return False

        self.updateBenefit(1)
        return True

    def findBallsBlockingHole(self, unpocketed):
        clean = cv2.imread(self.imgPath + "pooltable.png", 1)

        angle = utils.findAngle(self.targetPoint, self.ball.center, (self.ball.center[0], self.targetPoint[1]))
        
        rise, run, slope = utils.findRiseRunSlope(self.targetPoint, self.ball.center)
        if slope != 0:
            if slope < 0:
                angle = -angle
        else:
            if rise == 0:
                angle = -90

        self.hole.rotatedCenter = utils.rotateAround(self.ball.center, self.targetPoint, angle)

        tempPoints = []
        for n, b in unpocketed.items():
            if b.name == self.ball.name:
                continue
            else:
                b.rotatedCenter = utils.rotateAround(self.ball.center, b.center, angle)
                tempPoints.append(b)

        #bounding box
        minX = min(self.hole.rotatedCenter[0] - 10, self.hole.rotatedCenter[0] + 10, self.ball.center[0] - 10, self.ball.center[0] + 10)
        maxX = max(self.hole.rotatedCenter[0] - 10, self.hole.rotatedCenter[0] + 10, self.ball.center[0] - 10, self.ball.center[0] + 10)
        minY = min(self.hole.rotatedCenter[1], self.hole.rotatedCenter[1], self.ball.center[1], self.ball.center[1])
        maxY = max(self.hole.rotatedCenter[1], self.hole.rotatedCenter[1], self.ball.center[1], self.ball.center[1])

        t1, t2 = utils.findPointsOnEitherSideOf(self.ball.center, 10, -run, rise)
        t3, t4 = utils.findPointsOnEitherSideOf(self.targetPoint, 10, -run, rise)

        #checking for balls blocking
        blocking = []
        for b in tempPoints:
            if b.name != self.ball.name:
                cv2.circle(clean, utils.tupleToInt(b.center), 9,(255, 0, 0), 2)

                distance1 = utils.measureDistance(b.rotatedCenter, (minX, utils.clamp(b.rotatedCenter[1], minY, maxY)))
                distance2 = utils.measureDistance(b.rotatedCenter, (maxX, utils.clamp(b.rotatedCenter[1], minY, maxY)))
                distance3 = utils.measureDistance(b.rotatedCenter, (utils.clamp(b.rotatedCenter[0], minX, maxX), minY) )
                distance4 = utils.measureDistance(b.rotatedCenter, (utils.clamp(b.rotatedCenter[0], minX, maxX), maxY) )

                if distance1 <= 10 or distance2 <= 10 or distance3 <= 10 or distance4 <= 10:
                    cv2.circle(clean, utils.tupleToInt(b.center), 9,(255, 0, 255), 2)

                    blocking.append(b)

        cv2.line(clean, utils.tupleToInt(t1), utils.tupleToInt(t2),(255, 255, 255), 1)
        cv2.line(clean, utils.tupleToInt(t1), utils.tupleToInt(t3),(255, 255, 255), 1)

        cv2.line(clean, utils.tupleToInt(t2), utils.tupleToInt(t4),(255, 255, 255), 1)
        cv2.line(clean, utils.tupleToInt(t3), utils.tupleToInt(t4),(255, 255, 255), 1)

        # cv2.imshow('test', clean) 
        # cv2.waitKey(0)
        # cv2.destroyAllWindows() 

        return blocking
        
    def checkPathofCue(self, unpocketed):
        self.blockingCueToBall = self.findBallsBlockingCue(unpocketed)
        if len(self.blockingCueToBall) > 0:
            self.updateDifficulty(len(self.blockingCueToBall))

            return False

        self.updateBenefit(1)
        return True

    def findBallsBlockingCue(self, unpocketed):
        clean = cv2.imread(self.imgPath + "pooltable.png", 1)

        angle = utils.findAngle(self.cueball.center, self.hitPoint, (self.hitPoint[0], self.cueball.center[1]))
        if self.slopeToHP < 0:
            angle = -angle

        self.cueball.rotatedCenter = utils.rotateAround(self.hitPoint, self.cueball.center, angle)
 
        tempPoints = []
        for n, b in unpocketed.items():
            b.rotatedCenter = utils.rotateAround(self.hitPoint, b.center, angle)
            tempPoints.append(b)

        # bounding box of rotated points
        minX = min(self.cueball.rotatedCenter[0] - 10, self.cueball.rotatedCenter[0] + 10, self.hitPoint[0] - 10, self.hitPoint[0] + 10)
        maxX = max(self.cueball.rotatedCenter[0] - 10, self.cueball.rotatedCenter[0] + 10, self.hitPoint[0] - 10, self.hitPoint[0] + 10)
        minY = min(self.cueball.rotatedCenter[1], self.cueball.rotatedCenter[1], self.hitPoint[1], self.hitPoint[1])
        maxY = max(self.cueball.rotatedCenter[1], self.cueball.rotatedCenter[1], self.hitPoint[1], self.hitPoint[1])

        t1, t2 = utils.findPointsOnEitherSideOf(self.hitPoint, 10, -self.slopeToHPRun, self.slopeToHPRise)
        t3, t4 = utils.findPointsOnEitherSideOf(self.cueball.center, 10, -self.slopeToHPRun, self.slopeToHPRise)

        blocking = []
        for b in tempPoints:
            cv2.circle(clean, utils.tupleToInt(b.center), 9,(255, 0, 0), 2)
            
            distance1 = utils.measureDistance(b.rotatedCenter, (minX, utils.clamp(b.rotatedCenter[1], minY, maxY)))
            distance2 = utils.measureDistance(b.rotatedCenter, (maxX, utils.clamp(b.rotatedCenter[1], minY, maxY)))
            distance3 = utils.measureDistance(b.rotatedCenter, (utils.clamp(b.rotatedCenter[0], minX, maxX), minY) )
            distance4 = utils.measureDistance(b.rotatedCenter, (utils.clamp(b.rotatedCenter[0], minX, maxX), maxY) )

            if distance1 <= 10 or distance2 <= 10 or distance3 <= 10 or distance4 <= 10:
                cv2.circle(clean, utils.tupleToInt(b.center), 9,(255, 0, 255), 2)

                blocking.append(b)

        cv2.line(clean, utils.tupleToInt(t1), utils.tupleToInt(t2),(255, 255, 255), 1)
        cv2.line(clean, utils.tupleToInt(t2), utils.tupleToInt(t4),(255, 255, 255), 1)
        cv2.line(clean, utils.tupleToInt(t4), utils.tupleToInt(t3),(255, 255, 255), 1)
        cv2.line(clean, utils.tupleToInt(t3), utils.tupleToInt(t1),(255, 255, 255), 1)
        cv2.line(clean, utils.tupleToInt(self.hitPoint), utils.tupleToInt(self.targetPoint),(255, 0, 255), 1)

        # result = self.rotate_bound(clean, angle)
        # cv2.imwrite(f"{self.imgPath}\paths\\cue-ball\\rotated\\{self.cueball.name}-{self.ball.name}_rotated.png", result)

        # cv2.imwrite(f"{self.imgPath}\paths\\cue-ball\\{self.cueball.name}-{self.ball.name}_path.png", clean)

        # cv2.imshow('test', clean) 
        # # cv2.imshow('rotated', result) 
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()  

        return blocking

    def findAlternatePathToHole(self):
        pass

    def findAlternatePathToCue(self):
        pass