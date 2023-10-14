import logging
import traceback

import math
import os
import random
import time

import threading

import cv2
import imutils
import numpy as np

from pathlib import Path

from ball import Ball, BallCluster
from path import BallPath
import path
from utils import Line, OrEvent
from constants import constants

logger = logging.getLogger("ateball.round")

class Round(threading.Thread):

    def __init__(self, ipc, data, *args, **kwargs):   
        super().__init__(*args, **kwargs)

        self.ipc = ipc

        self.regions = constants.regions

        self.gamemode_info = data["gamemode"]["info"]
        self.gamemode_rules = data["gamemode"]["rules"]

        # round constants
        self.table = data["table"]
        
        self.suit = data['suit']
        self.turn_num = data['turn_num']
        self.round_path = str(Path(data["path"], f"round-{self.turn_num}"))
        self.images = { 
            "game" : self.table.images["original"],
            "table" : None,
            "pocketed" : None,
            "targets_bot" : None,
            "targets_opponent" : None,
        }
        #round constants

        self.ball_clusters = []
        self.in_cluster = {}
        self.break_required = False

        self.cueball = self.table.balls["cueball"] if "cueball" in self.table.balls else None

        self.targets = {}
        self.non_targets = {}

        self.viable_ball_paths =  []
        self.unviable_ball_paths =  []

        self.generate_viable_paths_complete = threading.Event()

        # start/stop by user/game
        self.round_start = threading.Event()
        self.round_stopped = threading.Event()

        # stop by exception or round completion
        self.round_complete = threading.Event()
        self.round_exception = threading.Event()
        self.round_over_event = OrEvent(self.round_complete, self.round_stopped, self.round_exception)

        self.logger = logging.getLogger("ateball.round")

    def run(self):
        try:
            start_time = time.time()

            os.makedirs(self.round_path, exist_ok=True)

            self.logger.info(f"Turn #{self.turn_num}")
            self.ipc.send_message({
                "type" : "ROUND-START", 
                "data" : {
                    "turn_num" : self.turn_num
                }
            })

            self.images["table"] = self.images["game"].copy()[self.regions.table[1]:self.regions.table[1]+self.regions.table[3], self.regions.table[0]:self.regions.table[0]+self.regions.table[2]]
            self.images["targets_bot"] = self.images["game"].copy()[self.regions.targets_bot[1]:self.regions.targets_bot[1]+self.regions.targets_bot[3], self.regions.targets_bot[0]:self.regions.targets_bot[0]+self.regions.targets_bot[2]]
            self.images["targets_opponent"] = self.images["game"].copy()[self.regions.targets_opponent[1]:self.regions.targets_opponent[1]+self.regions.targets_opponent[3], self.regions.targets_opponent[0]:self.regions.targets_opponent[0]+self.regions.targets_opponent[2]]
            self.images["targets_pocketed"] = self.images["game"].copy()[self.regions.targets_pocketed[1]:self.regions.targets_pocketed[1]+self.regions.targets_pocketed[3], self.regions.targets_pocketed[0]:self.regions.targets_pocketed[0]+self.regions.targets_pocketed[2]]

            cv2.imwrite(str(Path(self.round_path, "round_start.png")), self.images["game"])
            cv2.imwrite(str(Path(self.round_path, "table.png")), self.images["table"])
            cv2.imwrite(str(Path(self.round_path, "targets_pocketed.png")), self.images["targets_pocketed"])
            cv2.imwrite(str(Path(self.round_path, "targets_bot.png")), self.images["targets_bot"])
            cv2.imwrite(str(Path(self.round_path, "targets_opponent.png")), self.images["targets_opponent"])

            if self.gamemode_rules.ball_break:
                self.determine_ball_clusters()

            self.generate_viable_paths()

            self.round_over_event.notify(self.round_complete)
        except Exception as e:
            self.logger.error(traceback.format_exc())
            self.round_over_event.notify(self.round_exception)
        finally:
            self.logger.info("ROUND COMPLETE")
            self.ipc.send_message({"type" : "ROUND-COMPLETE"})

    def stop(self):
        self.logger.debug("stopping current round")
        cv2.destroyAllWindows()
        self.round_over_event.notify(self.round_stopped)

### Obtain shot on ball ###

    def determine_ball_clusters(self):
        self.logger.debug("Looking for ball clusters")

        # calculate distances between ball neighbors
        for n, b in self.table.hittable_balls.items():
            for n1, b1 in self.table.hittable_balls.items():
                if n == n1 or n in b1.neighbors:
                    continue
                
                dist = b.distance(b1)
                b.neighbors[n1] = dist
                b1.neighbors[n] = dist

        # calculate clusters of balls
        for n, b in self.table.hittable_balls.items():
            if n in self.in_cluster:
                continue

            closest_neighbors = b.get_closest_neighbors(self.table.hittable_balls)
            if not closest_neighbors:
                continue
            
            cluster = {n:b, **closest_neighbors}

            index = 1
            while not self.round_over_event.is_set():
                try:
                    b1 = cluster[next(iter(list(cluster)[index:]))]
                    cluster = {**cluster, **b1.get_closest_neighbors(self.table.hittable_balls, cluster)}
                    self.in_cluster = {**self.in_cluster, **cluster}

                    index+=1
                except StopIteration:
                    break
                
            self.ball_clusters.append(BallCluster(cluster))

            # break if cluster contains all
            if len(cluster) == len(self.table.hittable_balls):
                self.break_required = True
                break
        
        # create dict of cluster_identifier : bounds - identifier created by the summation of ord(combined names of balls in cluster)
        ball_clusters = { sum(map(ord, "".join([ b for b in cluster.balls ] ))) : cluster for cluster in self.ball_clusters }
        self.ipc.send_message({
            "type" : "ROUND-UPDATE", 
            "data" : {
                "type" : "SET-BALL-CLUSTERS",
                "ball_clusters" : ball_clusters
            }
        })

    def generate_viable_paths(self):
        self.logger.info("Generating viable shots...")

        try:
            start_time = time.time()

            self.cueball.get_closest_neighbors(self.table.hittable_balls, self.in_cluster)

            # generate dict of targets
            self.targets = self.table.hittable_balls.copy() if self.suit is None else { n:b for n,b in self.table.hittable_balls.items() if b.suit == self.suit }
            self.non_targets = {} if self.suit is None else { n:b for n,b in self.table.hittable_balls.items() if b.suit != self.suit }

            # sort targets by number if gamemode calls for it
            if self.gamemode_rules.order.inorder:
                self.targets = { n:b for n,b in sorted(self.targets.items(), key=lambda nb: nb[1].number)}

            self.logger.info(f"Determined targets: {[n for n, b in self.targets.items()]}")
            self.logger.info(f"Determined non-targets: {[n for n, b in self.non_targets.items()]}")

            # target ball clusters
            # for cluster in self.ball_clusters:
            #     self.logger.debug(list(cluster.balls))

            # sort targets in order of out-of-cluster to in-cluster
            sorted_targets = { n:b for n,b in sorted(self.targets.items(), key=lambda nb: nb[0] in self.in_cluster)}

            # testing
            color_order = {
                "yellow" : 1,
                "lightred" : 2,
                "darkred" : 3,
                "blue" : 4,
                "purple" : 5,
                "green" : 6,
                "orange" : 7,
                "eightball" : 8,
            }
            sorted_targets = { n:b for n,b in sorted(sorted_targets.items(), key=lambda nb: color_order[nb[1].color])}
            
            # check if there is AT LEAST one target available 
            if not sorted_targets:
                self.logger.debug(f"could not generate any shots (sorted_targets is empty) - {sorted_targets}")
                return

            sorted_target_indexes = [ n for n in sorted_targets ]
            index = 0

            test = self.images["table"].copy()
            
            while not self.generate_viable_paths_complete.is_set() and not self.round_over_event.is_set():
                try:
                    b = sorted_targets[sorted_target_indexes[index]]
                    # self.logger.debug(sorted_target_indexes[index])

                    if len(sorted_targets) > 1:
                        # skip eightball unless its the last ball
                        if b.name == "eightball" and self.gamemode_rules.order.inorder:
                            index += 1
                            continue

                    # find eligible path from ball to hole and sort paths by score
                    ball_paths = self.find_eligible_paths(b)
                    ball_paths = sorted(ball_paths, key=lambda p: p.difficulty, reverse=True)

                    # confirm eligible path from cueball to blal and sort by score
                    viable_ball_paths, unviable_ball_paths = self.confirm_path_eligibility(test, ball_paths)
                    viable_ball_paths = sorted(viable_ball_paths, key=lambda p: p.difficulty, reverse=True)

                    self.viable_ball_paths = [*self.viable_ball_paths, *viable_ball_paths]
                    self.unviable_ball_paths = [*self.unviable_ball_paths, *unviable_ball_paths]

                    if index == (len(sorted_target_indexes) - 1):
                        self.generate_viable_paths_complete.set()
                    index += 1
                except IndexError as e:
                    break

            self.ipc.send_message({
                "type" : "ROUND-UPDATE", 
                "data" : {
                        "type" : "SET-BALL-PATHS",
                        "ball_paths" : { bp.get_uuid() : bp for bp in self.viable_ball_paths }
                    }
            })

            for p in self.viable_ball_paths:
                p.draw(test)
                cue_to_ball_angle = p.target_ball_point.get_angle(p.cueball, p.target_hole.hole_gap_center)
                self.logger.debug(f"{p.target_ball.name} - {cue_to_ball_angle}")
                self.logger.debug(f"{constants.ball.entry_angle_bound.min < cue_to_ball_angle < constants.ball.entry_angle_bound.max}")
                cv2.imshow("Test", test)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            end_time = time.time() - start_time

            self.logger.debug(f"{len(self.viable_ball_paths)} - {self.viable_ball_paths}")
            self.logger.info(f"time to find paths: {end_time} seconds.")
        except Exception as e:
            self.logger.error(traceback.format_exc())

    def queryReboundedShot(self, ball):
        pass

### Obtain shot on ball ###


### Check paths ###

    def find_eligible_paths(self, ball):
        ball_paths = []

        try:
            start = time.time()

            # sort holes by whether or not they can be target - enough room for a ball?
            # goes here
            sorted_holes = self.table.holes

            for hole in sorted_holes:
                if hole.is_obscured_by_ball and hole.obscuring_ball != ball:
                    # possibily temporary
                    continue
                
                # sort holes by distance, clear path, direct vs bounce, etc.
                viable_paths = self.get_viable_paths(hole, ball)
                if hole.is_obscured_by_ball and hole.obscuring_ball == ball:
                    self.logger.debug(f"hole is obscured by {ball.name}")
                    ball_paths = viable_paths
                    break
                else:
                    ball_paths = [*ball_paths, *viable_paths]

            # self.logger.debug(f"found eligible shots {len(ball_paths)} - {time.time() - start} seconds")
        except Exception as e:
            self.logger.error(traceback.format_exc())

        return ball_paths

    def get_viable_paths(self, hole, ball):
        # describes if ball is so far inside pocket that there is no other option
        ball_inside_pocket = hole.inside_pocket(ball)
        hole_center = hole.center if not ball_inside_pocket else hole.corner

        # get ratio based on angle of entry - compensate left/right hole bounds based on ratio
        # if ball is left of hole center, left hole bound will be closer to center and right bound will be further, vice versa
        entry_angle = hole_center.get_angle(ball, hole.midpoint_table_intersection)
        entry_angle = math.fabs(360 - entry_angle if entry_angle > 180 else entry_angle)
        entry_angle_ratio = 1 - (entry_angle / hole.max_angle_of_entry)

        adjusted_shot_bounds_length = hole.shot_bounds_length * entry_angle_ratio

        on_left = ball.is_on_left(hole.outer_left, hole.inner_left) if hole.center.center[1] > self.table.table_center.center[1] else ball.is_on_left(hole.inner_left, hole.outer_left)
        if on_left:
            p1 = hole_center.find_points_along_slope(-adjusted_shot_bounds_length, hole.h_dy, hole.h_dx)
            p2 = hole_center.find_points_along_slope(hole.shot_bounds_length, hole.h_dy, hole.h_dx)
        else:
            p1 = hole_center.find_points_along_slope(-(hole.shot_bounds_length), hole.h_dy, hole.h_dx)
            p2 = hole_center.find_points_along_slope(adjusted_shot_bounds_length, hole.h_dy, hole.h_dx)

        # min bound determined by furtherst left, max by furthest right
        hole_bound_l, hole_bound_r = min(p1, p2, key=lambda p: p.center[0]), max(p1, p2, key=lambda p: p.center[0])
        if hole_center != hole_bound_l and hole_center != hole_bound_r:
            hole_target_bounds = [hole_center, hole_bound_l, hole_bound_r]
        else:
            hole_target_bounds = [hole_center, hole_bound_r] if hole_center == hole_bound_l else [hole_center, hole_bound_l]

        paths = []

        index = -1
        while not self.round_over_event.is_set():
            try:
                index += 1

                test = self.table.images["table"].copy() # testing
                
                hole_target_point = hole_target_bounds[index]

                ball_path = BallPath(self.cueball, ball, hole, hole_target_point, ball_inside_pocket)
                is_clear = ball_path.is_ball_trajectory_clear(self.table.hittable_balls)

                if not is_clear or ball_path.obscuring_ball_to_hole_trajectory:
                    # cannot be direct - must rebound or not possible
                    continue

                paths.append(ball_path)

            except IndexError as e:
                break

        return paths

    def confirm_path_eligibility(self, image, _ball_paths):
        viable_ball_paths = []
        unviable_ball_paths = []

        try:
            start = time.time()

            for ball_path in _ball_paths:
                is_clear = ball_path.is_cueball_trajectory_clear(image,  self.table.hittable_balls)

                if not is_clear or ball_path.obscuring_cue_to_ball_trajectory:
                    # cannot be direct - must rebound or not possible
                    unviable_ball_paths.append(ball_path)
                    continue

                viable_ball_paths.append(ball_path)

            # self.logger.debug(f"create shots {time.time() - start} seconds")
        except Exception as e:
            self.logger.error(traceback.format_exc())

        return viable_ball_paths, unviable_ball_paths
    

### Check paths ###


### Drawing ###

    def savePic(self):
        cv2.imwrite(self.round_path + "pooltableOutlined.png", self.round_image)
### Drawing ###


### Hit Ball ###

    def hitBall(self, ball):
        self.logger.debug("Hitting ball.")

        self.powerAmount(ball)

        print(ball.slopeToHPRun, ball.slopeToHPRise)
        if ball.slopeToHPRun > 0:
            print(1)
            if ball.slopeToHPRise > 0:
                print("1a")
                ball.hitPointStart = (ball.hitPointStart[0] + 2, ball.hitPointStart[1] + 2)
            else:
                print("1b")
                ball.hitPointStart = (ball.hitPointStart[0] - 2, ball.hitPointStart[1] - 2)
        else:
            print(2)
            if ball.slopeToHPRise > 0:
                print("2a")
                ball.hitPointStart = (ball.hitPointStart[0] - 2, ball.hitPointStart[1] + 2)
            else:
                print("2b")
                ball.hitPointStart = (ball.hitPointStart[0] + 2, ball.hitPointStart[1] + 3)

        print(ball.hitPoint, ball.hitPointStart, ball.hitPointEnd)
        pyautogui.moveTo((ball.hitPointStart[0] + self.regions.table_offset[0], ball.hitPointStart[1] + self.regions.table_offset[1]))
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pyautogui.moveTo((ball.hitPointStart[0] + self.regions.table_offset[0], ball.hitPointStart[1] + self.regions.table_offset[1]))
            pyautogui.mouseDown(button="left")
            pyautogui.moveTo((ball.hitPointEnd[0] + self.regions.table_offset[0]), (ball.hitPointEnd[1] + self.regions.table_offset[1]))
            pyautogui.dragTo((ball.hitPointEnd[0] + self.regions.table_offset[0], ball.hitPointEnd[1] + self.regions.table_offset[1]),
                             button="left", duration=1)
        
        pyautogui.screenshot(self.round_path + "confirmation.png",
                             region=self.regions.table)

        cv2.circle(self.round_image, utils.PointHelper.tupleToInt(ball.hitPoint), 3, (0, 0, 255), -1)
        cv2.circle(self.round_image, utils.PointHelper.tupleToInt(ball.hitPointStart), 3, (0, 0, 255), -1)
        cv2.circle(self.round_image, utils.PointHelper.tupleToInt(ball.hitPointEnd), 2, (0, 100, 255),-1)
        self.savePic()

    def powerAmount(self, ball):
        maxDistance = 1384
        pullDistance = 264
        velocity = 2768

        ratioCue = ball.distancetoCue / 664.0
        ratioBall = ball.distanceToHole / 664.0
        ratio = ratioCue + ratioBall
        if ratio < 1.0:
            distance = (ratio * 336) * 1.5
        else:
            distance = 336

        velocity = ratio * velocity

        print(ratioCue, ratioBall, ratio, velocity)


        if ball.slopeToHP > 0:
            if ball.slopeToHPRise > 0:
                #print(111)
                x = (ball.hitPointStart[0] + (distance * math.sqrt(1 / (1 + ball.slopeToHP ** 2))))
                y = (ball.hitPointStart[1] + ((ball.slopeToHP * distance) * math.sqrt( 1 / (1 + ball.slopeToHP ** 2))))
            else:
                #print(222)
                x = (ball.hitPointStart[0] - (distance * math.sqrt(1 / (1 + ball.slopeToHP ** 2))))
                y = (ball.hitPointStart[1] - ((ball.slopeToHP * distance) * math.sqrt(1 / (1 + ball.slopeToHP ** 2))))

            ball.hitPointEnd = (x, y)
        elif ball.slopeToHP < 0:
            if ball.slopeToHPRise > 0:
                #print(333)
                x = (ball.hitPointStart[0] - (distance * math.sqrt(1 / (1 + ball.slopeToHP ** 2))))
                y = (ball.hitPointStart[1] - ((ball.slopeToHP * distance) * math.sqrt(1 / (1 + ball.slopeToHP ** 2))))
            else:
                #print(444)
                x = (ball.hitPointStart[0] + (distance * math.sqrt(1 / (1 + ball.slopeToHP ** 2))))
                y = (ball.hitPointStart[1] + ((ball.slopeToHP * distance) * math.sqrt(1 / (1 + ball.slopeToHP ** 2))))

            ball.hitPointEnd = (x, y)
        else:
            if ball.slopeToHPRise == 0:
                #print(555)
                x = (ball.hitPointStart[0] + distance)
                y = ball.hitPointStart[1]
            else:
                #print(666)
                x = ball.hitPointStart[0]
                y = (ball.hitPointStart[1] - distance)

            ball.hitPointEnd = (x, y)
        #ball.stats()

### Hit Ball ###