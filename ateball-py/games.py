import logging
import traceback

import threading

import os
import cv2
import numpy as np

import json
from types import SimpleNamespace
from pathlib import Path
import base64

from abc import ABC

#files
from round import Round
from ball import Ball

import utils
import constants

class Game(threading.Thread, ABC):
    location: str
    img_game_start: str
    
    def __init__(self, ipc, location, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ipc = ipc

        self.name = self.__class__.__name__
        self.location = constants.locations[location] if location != "" else location

        self.game_constants = None
        with open(str(Path("ateball-py", "game_constants.json")), encoding='utf-8') as json_data:
            self.game_constants = json.load(json_data, object_hook=lambda d: SimpleNamespace(**d))

        self.game_start = threading.Event()
        self.game_cancelled = threading.Event()

        self.game_end = threading.Event()
        self.game_exception = threading.Event()
        self.game_over_event = utils.OrEvent(self.game_end, self.game_cancelled, self.game_exception)

        self.game_path = None
        self.game_num = 0

        self.window_capturer = utils.WindowCapturer(self.game_constants.regions.game, self.game_constants.regions.window_offset, 30, daemon=True)
        
        self.ball_locations = []

        self.suit = None
        self.turn_num = 0
        self.round_start_image = None

        self.player_turn = threading.Event()
        self.opponent_turn = threading.Event()
        self.turn_start = threading.Event()
        self.turn_start_event = utils.OrEvent(self.turn_start, self.game_over_event)

        self.current_round = None

        self.logger = logging.getLogger("ateball.games")

    def run(self):
        try:
            self.logger.info(f"Playing {self.name}...")

            self.window_capturer.start()

            self.wait_for_game_start()
            if self.game_start.is_set():
                self.game_path = str(Path("ateball-py", "games", f"{self.game_num}-{self.name}-{self.location}"))
                os.makedirs(self.game_path, exist_ok=True)

                self.window_capturer.record(self.game_path, "game")

                self.ipc.send_message({"type" : "GAME-START"})
                self.logger.info(f"\nGame #{self.game_num}\n")
                
                threading.Thread(target=self.get_ball_locations, daemon=True).start()
                threading.Thread(target=self.wait_for_turn_start, daemon=True).start()

                while not self.game_over_event.is_set():
                    self.turn_start_event.wait()
                    if self.player_turn.is_set() and self.turn_start.is_set():
                        self.turn_start.clear()

                        self.ipc.send_message({"type" : "ROUND-START"})

                        round_path = Path(self.game_path, f"round-{self.turn_num}")
                        os.makedirs(round_path, exist_ok=True)

                        cv2.imwrite(str(Path(round_path, "round_start.png")), self.round_start_image)

                        self.logger.info(f"Turn #{self.turn_num}")
                        
                        round_data = self.get_round_data(round_path)
                        self.current_round = Round(round_data, self.game_constants)
                        result = self.current_round.start()
        except Exception as e:
            self.logger.error(traceback.format_exc())
        finally:
            self.window_capturer.stop()

            self.game_over_event.notify(self.game_end)
            if self.game_cancelled.is_set():
                self.ipc.send_message({"type" : "GAME-CANCELLED"})
            elif self.game_end.is_set():
                self.ipc.send_message({"type" : "GAME-END"})

    def cancel(self):
        self.logger.debug("cancelling current game")
        self.game_over_event.notify(self.game_cancelled)
        self.game_over_event.wait()

    def wait_for_game_start(self):
        self.logger.info("Waiting for game to start...")

        # get contour of marker
        needle = cv2.imread(utils.ImageHelper.imagePath(self.img_game_start))
        needle_contours = self.get_game_marker_contours(needle)

        while not self.game_start.is_set() and not self.game_cancelled.is_set() and not self.game_over_event.is_set():
            try:
                image = self.window_capturer.get_first()
                if image.any() and needle_contours.any():
                    # get contour of (potential) marker of current game
                    haystack = utils.CV2Helper.slice_image(image, self.game_constants.regions.turn_start)
                    haystack_contours = self.get_game_marker_contours(haystack)
                    
                    # match shape of contours - contours similar closer to 0
                    match = cv2.matchShapes(haystack_contours, needle_contours, cv2.CONTOURS_MATCH_I1 , 0.0)
                    if match <= .05:
                        self.get_game_num()
                        self.game_start.set()
            except Exception as e:
                self.logger.error(traceback.format_exc())

    def get_game_marker_contours(self, image):
        try:
            # resize & blur to get clearer contours
            image = utils.CV2Helper.resize(image, 4)
            image_blur = cv2.GaussianBlur(image, (9,9), 0)

            image_hsv = cv2.cvtColor(image_blur, cv2.COLOR_BGR2HSV)

            # filter out gray background to get shape of marker
            image_mask = cv2.inRange(image_hsv, np.array([0, 0, 40]), np.array([180, 255, 255]))
            image_mask = cv2.morphologyEx(image_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=3)

            # return largest contour
            contours, hierarchy = cv2.findContours(image_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[0]
        except IndexError as e:
            return np.array([])
        else:
            return contours

    def get_game_num(self):
        json_path = Path("ateball-py", "game.json")
        json_path.touch(exist_ok=True)

        with open(json_path, "r") as f:
            try:
                data = json.load(f)
                data["num"] += 1 
                self.game_num = data["num"]
                with open(json_path, "w") as f:
                    f.write(json.dumps(data))
            except json.decoder.JSONDecodeError as e:
                data = {"num" : 1}
                self.game_num = data["num"]
                with open(json_path, "w") as f:
                    f.write(json.dumps(data))

    def wait_for_turn_start(self):
        self.logger.info("Waiting for turn to start...")

        # load turn timer masks
        turn_timer_mask = cv2.imread(utils.ImageHelper.imagePath(constants.img_turn_timers_mask), 0)
        turn_timer_mask_single =cv2.imread(utils.ImageHelper.imagePath(constants.img_turn_timers_mask_single), 0)

        old_player_status = None
        while not self.game_over_event.is_set():
            try:
                image = self.window_capturer.get_first()
                if image.any():
                    # get mean color of each turn timer - colors represent different state at beginning of round
                    turn_status = self.get_turn_status(image, turn_timer_mask, turn_timer_mask_single)

                    # get status of player/opponent turn timer - color, open/closed, timeout
                    player_status, opponent_status = turn_status["player"]["status"], turn_status["opponent"]["status"]
                    player_timer_start, opponent_timer_start = turn_status["player"]["hierarchy"], turn_status["opponent"]["hierarchy"]
                    timed_out = ((player_status == "timeout" and player_timer_start) or (opponent_status == "timeout" and opponent_timer_start))

                    if self.player_turn.is_set():
                        if opponent_status != "pending":
                            if opponent_status == "successive":
                                # on successive pockets, player turn starts when player timer flashes white and contour is 'closed'
                                # *as long as player status in the previous frame is not 'started'
                                if (player_status == "started" and player_timer_start) and old_player_status != "started":
                                    self.end_existing_round()

                                    self.turn_num += 1
                                    self.set_round_image(timed_out)
                                    self.player_turn.set()
                                    self.turn_start_event.notify(self.turn_start)
                            else:
                                # player turn ends when opponent timer flashes white and contour is 'closed'
                                if opponent_status == "started" and opponent_timer_start:
                                    self.end_existing_round()

                                    self.set_round_image(timed_out)
                                    self.opponent_turn.set()
                    elif self.opponent_turn.is_set():
                        # player turn starts when player timer flashes white and contour is 'closed'
                        if player_status == "started" and player_timer_start:
                            self.end_existing_round()

                            self.turn_num += 1
                            self.set_round_image(timed_out)
                            self.player_turn.set()
                            self.turn_start_event.notify(self.turn_start)
                    else:
                        # player and opponent turn will both not be set at start of game

                        if player_status != "pending":
                            self.end_existing_round()

                            self.turn_num += 1
                            self.set_round_image(timed_out)
                            self.player_turn.set()
                            self.turn_start_event.notify(self.turn_start)
                        else:
                            self.player_turn.clear()
                            self.opponent_turn.set()
            except Exception as e:
                self.logger.error(traceback.format_exc())
            finally:
                # keep track of last frame's player status
                old_player_status = player_status

    def get_turn_status(self, image, mask, mask_single):
        # status indicated by the color of turn timer and whether timer is 'open' or 'closed' (at the start of a turn)

        # slice specified image to size - match size of mask
        turn_timer = utils.CV2Helper.slice_image(image, self.game_constants.regions.turn_timer)

        # mask out background - show only turn timers
        turn_timers_masked = cv2.bitwise_and(turn_timer, turn_timer, mask=mask)

        # slice image to seperate timers
        p_timer = utils.CV2Helper.slice_image(turn_timers_masked, self.game_constants.regions.player_turn_timer)
        o_timer = utils.CV2Helper.slice_image(turn_timers_masked, self.game_constants.regions.opponent_turn_timer)

        # match mean color to closest color
        closest_color_player = utils.CV2Helper.get_closest_color(p_timer, mask_single, self.game_constants.turn_status.__dict__)
        closest_color_opponent = utils.CV2Helper.get_closest_color(o_timer, mask_single, self.game_constants.turn_status.__dict__)

        # get hierarchy of turn timer contours (closed vs open)
        p_timer_hsv = cv2.cvtColor(p_timer, cv2.COLOR_BGR2HSV)
        o_timer_hsv = cv2.cvtColor(o_timer, cv2.COLOR_BGR2HSV)

        p_hierarchy_mask = utils.CV2Helper.create_mask(p_timer_hsv, np.array([0, 0, 100]), np.array([180, 255, 255]), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        o_hierarchy_mask = utils.CV2Helper.create_mask(o_timer_hsv, np.array([0, 0, 100]), np.array([180, 255, 255]), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

        p_contours, p_hierarchy = cv2.findContours(p_hierarchy_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        o_contours, o_hierarchy = cv2.findContours(o_hierarchy_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        return {
            "player" : {
                "status" : closest_color_player,
                "hierarchy" : len(p_hierarchy[0]) > 1 if p_hierarchy is not None else False
            },
            "opponent" : {
                "status" : closest_color_opponent,
                "hierarchy" : len(o_hierarchy[0]) > 1 if o_hierarchy is not None else False
            },
        }

    def end_existing_round(self):
        self.opponent_turn.clear()
        self.player_turn.clear()

        self.ipc.send_message({"type" : "ROUND-END"})

        if self.current_round is not None and not self.current_round.round_over_event.is_set():
            self.current_round.round_over_event.notify(self.current_round.round_cancel)
            self.current_round = None
            self.logger.info(f"Turn #{self.turn_num} Complete")

        self.export_round_data()

    def set_round_image(self, turn_timed_out):
        if self.round_start_image is not None:
            # in case of timeout, use last recorded round_start_image
            self.round_start_image = self.round_start_image if turn_timed_out else self.window_capturer.get_last()
        else:
            self.round_start_image = self.window_capturer.get_last()

    def export_round_data(self):
        pass

    def get_round_data(self, image_path):
        return {
            "save_image_path" : image_path,
            "round_image" : self.round_start_image,
            "suit" : self.suit,
            "all_balls" : {},
            "unpocketed_balls" : [],
            "pocketed_balls" : {},
            "targets" : {},
            "nontargets" : {},
        }

    def is_game_over(self, image):
        pos = utils.ImageHelper.imageSearch(self.img_game_end, image, region=self.game_constants.regions.table)
        return True if pos else False

    def get_ball_locations(self):
        self.logger.debug("Outlining pool balls...\n")

        while not self.game_over_event.is_set():
            try:
                image = self.window_capturer.get_first()
                if image.any():
                    table = utils.CV2Helper.slice_image(image, constants.regions.table)
                    table_hsv = cv2.cvtColor(table, cv2.COLOR_BGR2HSV)

                    table_masked = self.mask_out_table(table, table_hsv)
                    table_masked_gray = cv2.cvtColor(table_masked, cv2.COLOR_BGR2GRAY)

                    # Hough Circles for identifying ball locations more accurate than contours
                    points = cv2.HoughCircles(table_masked_gray, cv2.HOUGH_GRADIENT, 1, 17, param1=20, param2=9, minRadius=9, maxRadius=11)
                    points = np.uint16(np.around(points))

                    self.ball_locations = [Ball((p[0], p[1])) for p in points[0, :]]

                    for b in self.ball_locations:
                        b.draw(table)

                    retval, image_buffer = cv2.imencode('.png', table)
                    image_buffer = base64.b64encode(image_buffer.tobytes()).decode('ascii')
                    image_b64 = f"data:image/png;base64,{image_buffer}"

                    self.ipc.send_message({"type" : "REALTIME-STREAM", "data" : image_b64})
            except Exception as e:
                self.logger.error(traceback.format_exc())

    def mask_out_table(self, img, hsv):
        # mask for table color - table color masked out for visibility
        blue_lower = np.array([90, 80, 0])
        blue_upper = np.array([106, 255, 255])
        black_lower = np.array([0, 0, 35])
        black_high = np.array([180, 255, 255])

        table_invert_mask = cv2.inRange(hsv, blue_lower, blue_upper)
        table_invert_mask = cv2.bitwise_not(table_invert_mask)

        hole_mask = cv2.inRange(hsv, black_lower, black_high)

        table_mask = cv2.bitwise_and(table_invert_mask, hole_mask)
        table_masked_out = cv2.bitwise_and(img, img, mask=table_mask)

        return table_masked_out

class ONE_ON_ONE(Game):
    def __init__(self, location, pipe, *args, **kwargs):
        super().__init__(location, pipe, *args, **kwargs)

        self.img_game_start = "bet_marker.png"

class TOURNAMENT(Game):
    def __init__(self, location, pipe, *args, **kwargs):
        super().__init__(location, pipe, *args, **kwargs)

        self.img_game_start = "bet_marker.png"

class NO_GUIDELINE(Game):
    def __init__(self, location, pipe, *args, **kwargs):
        super().__init__(location, pipe, *args, **kwargs)

        self.img_game_start = "bet_marker.png"

class NINE_BALL(Game):
    def __init__(self, location, pipe, *args, **kwargs):
        super().__init__(location, pipe, *args, **kwargs)

        self.img_game_start = "bet_marker.png"

class LUCKY_SHOT(Game):
    def __init__(self, location, pipe, *args, **kwargs):
        super().__init__(location, pipe, *args, **kwargs)

        self.img_game_start = "lucky_shot.png"

class CHALLENGE(Game):
    def __init__(self, location, pipe, *args, **kwargs):
        super().__init__(location, pipe, *args, **kwargs)

        self.img_game_start = "bet_marker.png"

class PASS_N_PLAY(Game):
    def __init__(self, location, pipe, *args, **kwargs):
        super().__init__(location, pipe, *args, **kwargs)

        self.img_game_start = "practice_marker.png"

class QUICK_FIRE(Game):
    def __init__(self, location, pipe, *args, **kwargs):
        super().__init__(location, pipe, *args, **kwargs)

        self.img_game_start = "practice_marker.png"

class GUEST(Game):
    def __init__(self, location, *args, **kwargs):
        super().__init__(location, *args, **kwargs)

        self.img_game_start = "game\\bet_marker.png"