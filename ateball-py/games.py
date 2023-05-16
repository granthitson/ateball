import logging
import traceback

import threading
import queue as q

import os
import shutil
import cv2
import numpy as np

import json
from pathlib import Path
import base64

from abc import ABC

#files
from table import Table
from round import Round
from ball import Ball

import utils
from constants import constants

class Game(threading.Thread, ABC):
    location: str

    def __init__(self, ipc, location, realtime_config={}, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ipc = ipc

        self.name = self.__class__.__name__
        self.location = constants.locations.__dict__[location] if location else None

        self.gamemode_info = constants.gamemodes.__dict__[self.__class__.__name__]
        self.gamemode_rules = constants.rules.__dict__[self.gamemode_info.rules] if "rules" in self.gamemode_info.__dict__ else constants.rules.__dict__[self.location.rules]

        self.img_game_start = constants.gamemodes.__dict__[self.__class__.__name__].img_game_start

        self.game_start = threading.Event()
        self.game_cancelled = threading.Event()

        self.game_end = threading.Event()
        self.game_exception = threading.Event()
        self.game_over_event = utils.OrEvent(self.game_end, self.game_cancelled, self.game_exception)

        self.game_data = {
            "num" : 0,
            "path" : "",
            "suit" : None,
            "turn_num" : 0, 
            "table_data" : None,
            "round_image" : None
        }

        self.window_capturer = utils.WindowCapturer(constants.regions.game, constants.regions.window_offset, 30, daemon=True)
        self.recording = q.Queue()

        self.realtime_config = realtime_config
        self.realtime_update = threading.Event()
        
        self.table = Table(self.gamemode_info, self.game_data)
        self.table_history = []

        self.current_round = None

        self.logger = logging.getLogger("ateball.games")
        self.fhandler = None

    def configure_game_dir(self):
        if bool(os.getenv("GAME_DEBUG")):
            self.game_data["path"] = str(Path("ateball-py", "games", "DEBUG"))
            if os.path.exists(self.game_data["path"]):
                shutil.rmtree(self.game_data["path"])
        else:
            full_game_name = "-".join(str(d) for d in [self.game_data["num"], self.name, self.location] if d is not None)
            self.game_data["path"] = str(Path("ateball-py", "games", full_game_name))

        os.makedirs(self.game_data["path"], exist_ok=True)

    def configure_logging(self):
        formatter = utils.Formatter()

        self.fhandler = logging.FileHandler(f"{self.game_data['path']}/log.log", mode="w")
        self.fhandler.setFormatter(formatter)
        self.fhandler.setLevel(os.environ.get("LOG_LEVEL"))

        logging.getLogger("ateball").addHandler(self.fhandler)

    def clear_logging(self):
        logging.getLogger("ateball").removeHandler(self.fhandler)
        self.fhandler = None

    def record(self):
        # write images to avi file
        v_format = cv2.VideoWriter_fourcc(*'XVID')

        region = (constants.regions.table[2], constants.regions.table[3] * 2)
        video_writer = cv2.VideoWriter(str(Path(self.game_data["path"], "game.avi")), v_format, self.window_capturer.fps, region)

        while not self.game_over_event.is_set():
            try:
                table = self.recording.get()

                original = table.images["table"]
                empty = np.stack((table.images["none"],)*3, axis=-1)
                if not original.any():
                    break

                drawn = table.draw()
                
                stack = np.concatenate((original, drawn), axis=0)
                video_writer.write(stack)
            except Exception as e:
                self.logger.error(traceback.format_exc())
        
        video_writer.release()

    def run(self):
        raise NotImplementedError("must define run to use this base class")

    def wait_for_game_start(self):
        self.logger.info("Waiting for game to start...")

        # get contour of marker
        needle = utils.CV2Helper.imread(self.img_game_start)
        needle_contours = self.get_game_marker_contours(needle)

        while not self.game_start.is_set() and not self.game_over_event.is_set():
            try:
                image = self.window_capturer.get()
                if image.any() and needle_contours.any():
                    # get contour of (potential) marker of current game
                    haystack = utils.CV2Helper.slice_image(image, constants.regions.turn_start)
                    haystack_contours = self.get_game_marker_contours(haystack)
                    
                    # match shape of contours - contours similar closer to 0
                    match = cv2.matchShapes(haystack_contours, needle_contours, cv2.CONTOURS_MATCH_I1 , 0.0)
                    if match <= .02:
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
                self.game_data["num"] = data["num"]
                with open(json_path, "w") as f:
                    f.write(json.dumps(data))
            except json.decoder.JSONDecodeError as e:
                data = {"num" : 1}
                self.game_data["num"] = data["num"]
                with open(json_path, "w") as f:
                    f.write(json.dumps(data))

    def update_user_targets(self, data):
        self.table.user_targets = data["targets"]

    def cancel(self):
        self.logger.debug("cancelling current game")
        self.game_over_event.notify(self.game_cancelled)

    def is_game_over(self, image):
        pos = utils.CV2Helper.imageSearch(self.img_game_end, image, region=constants.regions.table)
        return True if pos else False

class OnePlayerGame(Game):
    def __init__(self, ipc, location, realtime_config={}, *args, **kwargs):
        pass

class TwoPlayerGame(Game):
    def __init__(self, ipc, location, realtime_config={}, *args, **kwargs):
        super().__init__(ipc, location, realtime_config, *args, **kwargs)

        self.player_turn = threading.Event()
        self.opponent_turn = threading.Event()

        self.turn_start = threading.Event()
        self.turn_end = threading.Event()
        self.turn_start_event = utils.OrEvent(self.turn_start, self.turn_end)
        self.timed_out = threading.Event()

    def run(self):
        try:
            self.logger.info(f"Playing {self.name}...")

            self.window_capturer.start()

            self.wait_for_game_start()
            if self.game_start.is_set():
                self.configure_game_dir()
                self.configure_logging()

                self.ipc.send_message(
                    {
                        "type" : "GAME-START", 
                        "data" : { 
                            "suit" : self.gamemode_rules.suit.choice,
                            "balls" : { b.name: b.get_state() for b in self.table.hittable_balls }
                        }
                    }
                )
                self.logger.info(f"Game #{self.game_data['num']}")

                threading.Thread(target=self.record).start()
                
                threading.Thread(target=self.wait_for_turn_start, daemon=True).start()

                # game loop
                while not self.game_over_event.is_set():
                    image = self.window_capturer.get()
                    if image.any():
                        self.table.identify_targets(image)

                        # keep history of image and table data
                        if len(self.table_history) >= 30:
                            self.table_history.pop(0)
                        self.table_history.append((image, self.table.copy()))

                        if self.turn_start.is_set():
                            # update round start image on turn start (play or opponent)
                            self.turn_start.clear()
                            self.set_current_round_data()

                            # update ball state on round by round basis
                            self.ipc.send_message(
                                {
                                    "type" : "UPDATE-BALL-STATE", 
                                    "data" : { 
                                        "balls" : { b.name: b.get_state() for b in self.table.hittable_balls }
                                    }
                                }
                            )
                            
                            if self.player_turn.is_set():
                                self.game_data["turn_num"] += 1

                                self.current_round = Round(self.ipc, self.game_data)
                                self.current_round.start()

                            self.timed_out.clear()

                        # draw table
                        drawn_image = self.table.draw(self.realtime_config)

                        # send updated table image if it differs from last
                        if self.table.updated.is_set() or self.realtime_update.is_set():
                            self.realtime_update.clear()

                            retval, image_buffer = cv2.imencode('.png', drawn_image)
                            image_buffer = base64.b64encode(image_buffer.tobytes()).decode('ascii')
                            image_b64 = f"data:image/png;base64,{image_buffer}"

                            self.ipc.send_message({"type" : "REALTIME-STREAM", "data" : image_b64})

                        self.recording.put(self.table.copy())

                self.recording.put((np.array([]), np.array([])))
        except Exception as e:
            self.logger.error(traceback.format_exc())

            self.game_over_event.notify(self.game_exception)
        finally:
            self.window_capturer.stop()
            self.clear_logging()

            if self.game_exception.is_set():
                self.ipc.send_message({"type" : "GAME-EXCEPTION"})
            else:
                self.game_over_event.notify(self.game_end)
                if self.game_cancelled.is_set():
                    self.ipc.send_message({"type" : "GAME-CANCELLED"})
                elif self.game_end.is_set():
                    self.ipc.send_message({"type" : "GAME-END"})

    def wait_for_turn_start(self):
        self.logger.info("Waiting for turn to start...")

        # load turn timer masks
        turn_timer_mask = utils.CV2Helper.imread(constants.images.img_turn_timers_mask, 0)
        turn_timer_mask_single = utils.CV2Helper.imread(constants.images.img_turn_timers_mask_single, 0)

        old_player_status, old_opponent_status = None, None
        while not self.game_over_event.is_set():
            try:
                image = self.window_capturer.get()
                if image.any():
                    # get mean color of each turn timer - colors represent different state at beginning of round
                    turn_status = self.get_turn_status(image, turn_timer_mask, turn_timer_mask_single)

                    # get status of player/opponent turn timer - color, open/closed, timeout
                    player_status, opponent_status = turn_status["player"]["status"], turn_status["opponent"]["status"]
                    player_timer_start, opponent_timer_start = turn_status["player"]["hierarchy"], turn_status["opponent"]["hierarchy"]
                    timed_out = ((player_status == "timeout" and player_timer_start) or (opponent_status == "timeout" and opponent_timer_start))

                    if timed_out:
                        self.timed_out.set()

                    if self.player_turn.is_set():
                        if opponent_status != "pending":
                            if opponent_status == "successive":
                                # on successive pockets, player turn starts when player timer flashes white and contour is 'closed'
                                # *as long as player status in the previous frame is not 'started'
                                if (player_status == "started" and player_timer_start) and old_player_status != "started":
                                    if not self.turn_start.is_set():
                                        self.end_existing_round()

                                        self.player_turn.set()
                                        self.turn_start_event.notify(self.turn_start)
                            else:
                                # player turn ends when opponent timer flashes white and contour is 'closed'
                                if opponent_status == "started" and opponent_timer_start:
                                    if not self.turn_start.is_set():
                                        self.end_existing_round()

                                        self.opponent_turn.set()
                                        self.turn_start_event.notify(self.turn_start)
                    elif self.opponent_turn.is_set():
                        if player_status != "pending":
                            if player_status == "successive":
                                # on successive pockets, opponent turn starts when opponent timer flashes white and contour is 'closed'
                                # *as long as opponent status in the previous frame is not 'started'
                                if (opponent_status == "started" and opponent_timer_start) and old_opponent_status != "started":
                                    if not self.turn_start.is_set():
                                        self.end_existing_round()

                                        self.opponent_turn.set()
                                        self.turn_start_event.notify(self.turn_start)
                            else:
                                # opponent turn ends when opponent timer flashes white and contour is 'closed'
                                if player_status == "started" and player_timer_start:
                                    if not self.turn_start.is_set():
                                        self.end_existing_round()

                                        self.player_turn.set()
                                        self.turn_start_event.notify(self.turn_start)
                    else:
                        # player and opponent turn will both not be set at start of game

                        if player_status != "pending":
                            if not self.turn_start.is_set():
                                self.end_existing_round()

                                self.opponent_turn.clear()
                                self.player_turn.set()
                                self.turn_start_event.notify(self.turn_start)
                        else:
                            if not self.turn_start.is_set():
                                self.player_turn.clear()
                                self.opponent_turn.set()
                                self.turn_start_event.notify(self.turn_start)
            except Exception as e:
                self.logger.error(traceback.format_exc())
            finally:
                # keep track of last frame's player status
                old_player_status = player_status
                old_opponent_status = opponent_status

    def get_turn_status(self, image, mask, mask_single):
        # status indicated by the color of turn timer and whether timer is 'open' or 'closed' (at the start of a turn)

        # slice specified image to size - match size of mask
        turn_timer = utils.CV2Helper.slice_image(image, constants.regions.turn_timer)

        # mask out background - show only turn timers
        turn_timers_masked = cv2.bitwise_and(turn_timer, turn_timer, mask=mask)

        # slice image to seperate timers
        p_timer = utils.CV2Helper.slice_image(turn_timers_masked, constants.regions.player_turn_timer)
        o_timer = utils.CV2Helper.slice_image(turn_timers_masked, constants.regions.opponent_turn_timer)

        # match mean color to closest color
        closest_color_player = utils.CV2Helper.get_closest_color(p_timer, mask_single, constants.turn.status.__dict__)
        closest_color_opponent = utils.CV2Helper.get_closest_color(o_timer, mask_single, constants.turn.status.__dict__)

        # get hierarchy of turn timer contours (closed vs open)
        p_timer_hsv = cv2.cvtColor(p_timer, cv2.COLOR_BGR2HSV)
        o_timer_hsv = cv2.cvtColor(o_timer, cv2.COLOR_BGR2HSV)

        turn_backround_mask = np.array(constants.turn.background_mask.lower), np.array(constants.turn.background_mask.upper)
        p_hierarchy_mask = utils.CV2Helper.create_mask(p_timer_hsv, *turn_backround_mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        o_hierarchy_mask = utils.CV2Helper.create_mask(o_timer_hsv, *turn_backround_mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

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

    def set_current_round_data(self):
        if self.game_data["table_data"] is None:
            self.game_data["round_image"], self.game_data["table_data"] = self.table_history[0]
        else:
            if not self.timed_out.is_set():
                self.game_data["round_image"], self.game_data["table_data"] = self.table_history[0]

    def end_existing_round(self):
        self.opponent_turn.clear()
        self.player_turn.clear()

        self.ipc.send_message({"type" : "ROUND-END"})

        if self.current_round is not None and not self.current_round.round_over_event.is_set():
            self.current_round.round_over_event.notify(self.current_round.round_cancel)
            self.current_round = None
            self.logger.info("Turn #{} Complete".format(self.game_data["turn_num"]))

    def cancel(self):
        self.logger.debug("cancelling current game")
        self.turn_start_event.notify(self.turn_end)
        self.game_over_event.notify(self.game_cancelled)

class ONE_ON_ONE(TwoPlayerGame):
    def __init__(self, location, pipe, *args, **kwargs):
        super().__init__(location, pipe, *args, **kwargs)

class TOURNAMENT(TwoPlayerGame):
    def __init__(self, location, pipe, *args, **kwargs):
        super().__init__(location, pipe, *args, **kwargs)

class NO_GUIDELINE(TwoPlayerGame):
    def __init__(self, location, pipe, *args, **kwargs):
        super().__init__(location, pipe, *args, **kwargs)

class NINE_BALL(TwoPlayerGame):
    def __init__(self, location, pipe, *args, **kwargs):
        super().__init__(location, pipe, *args, **kwargs)

class LUCKY_SHOT(OnePlayerGame):
    def __init__(self, location, pipe, *args, **kwargs):
        super().__init__(location, pipe, *args, **kwargs)

class CHALLENGE(TwoPlayerGame):
    def __init__(self, location, pipe, *args, **kwargs):
        super().__init__(location, pipe, *args, **kwargs)

class PASS_N_PLAY(TwoPlayerGame):
    def __init__(self, location, pipe, *args, **kwargs):
        super().__init__(location, pipe, *args, **kwargs)

class QUICK_FIRE(OnePlayerGame):
    def __init__(self, location, pipe, *args, **kwargs):
        super().__init__(location, pipe, *args, **kwargs)

class GUEST(TwoPlayerGame):
    def __init__(self, location, *args, **kwargs):
        super().__init__(location, *args, **kwargs)