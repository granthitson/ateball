import logging
import traceback

import threading
import queue as q

#files
import games
import utils

class AteBall():
    def __init__(self):
        self.ipc = utils.IPC()

        self.processing_play_request = threading.Event()
        self.active_game = None

        self.quit_event = threading.Event()

        self.logger = logging.getLogger("ateball")

    def process_message(self):
        self.logger.info("waiting for command...")
        while not self.quit_event.is_set():
            try:
                response = None # change to object/dict to send response

                msg = self.ipc.incoming.get()

                m_type = msg["type"]
                if m_type == "play":
                    if self.processing_play_request.is_set() or (self.active_game and not self.active_game.game_over_event.is_set()):
                        response = { "type" : "BUSY"}
                    else:
                        if self.active_game and self.active_game.game_over_event.is_set():
                            self.active_game = None

                        threading.Thread(target=self.play, args=(msg,), daemon=True).start()
                elif m_type == "show-realtime":
                    if self.active_game:
                        if self.active_game.show_realtime_event.is_set():
                            self.active_game.show_realtime_event.clear()
                        else:
                            self.active_game.show_realtime_event.set()
                elif m_type == "realtime-configure":
                    if self.active_game:
                        self.active_game.realtime_config.update(msg["data"])
                        self.active_game.realtime_update.set()
                elif m_type == "select-ball-path":
                    if self.active_game and self.active_game.current_round:
                        self.active_game.current_round.select_ball_path(msg["data"])
                elif m_type == "update-targets":
                    if self.active_game:
                        self.active_game.update_user_targets(msg["data"])
                elif m_type == "cancel" or m_type == "stop":
                    self.stop()
                elif m_type == "quit":
                    self.quit()
                else:
                    pass
            except q.Empty:
                pass
            except KeyError as e:
                response["status"] = "failed"
                response["msg"] = str(e)
            except Exception as e:
                self.logger.error(f"error processing message : {traceback.format_exc()}")
            else: 
                if response is not None:
                    if (isinstance(response, dict)):
                        response["id"] = msg["id"]
                    response["status"] = "failed"

                    self.ipc.outgoing.put(response)


    ###initialization

    def start(self):
        try:
            self.logger.info("Starting Ateball...")

            threading.Thread(target=self.ipc.listen, daemon=True).start()
            threading.Thread(target=self.ipc.send, daemon=True).start()

            if self.ipc.listen_event.wait(5):
                #start receiving msgs and initialize
                threading.Thread(target=self.process_message, daemon=True).start()
        except Exception as e:
            self.logger.error(f"{type(e)} - {e}")
            self.quit()
        else:
            self.ipc.send_message({"type" : "INIT"})

    ###initialization

    ###menu

    def play(self, data):
        try:
            self.processing_play_request.set()

            game_config = data['game_config'] if "game_config" in data else {}
            realtime_config = data['realtime_config'] if "realtime_config" in data else {}
        
            location = game_config['location'] if "location" in game_config else None
        
            Game = getattr(games, game_config['gamemode'].upper())
            self.active_game = Game(self.ipc, location, realtime_config, daemon=True)
            self.active_game.start()
        except Exception as e:
            if isinstance(e, KeyError):
                self.logger.error(f"error playing game: invalid gamemode location - {data['game_config']['location']}")
            elif isinstance(e, AttributeError):
                self.logger.error(f"error playing game: invalid gamemode - {data['game_config']['gamemode']}")
            # else:
            self.logger.error(f"error playing game: {traceback.format_exc()}")

            self.ipc.send_message({"type" : "GAME-CANCELLED"})
        finally:
            self.processing_play_request.clear()

    def stop(self):
        if self.active_game:
            self.active_game.stop()
            self.active_game.join()
            self.active_game = None
            self.ipc.send_message({"type" : "GAME-STOPPED"})

    ###menu

    def quit(self):
        self.ipc.quit()
        self.quit_event.set()