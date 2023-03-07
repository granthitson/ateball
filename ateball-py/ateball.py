import logging
import traceback

import threading
import queue as q

#files
import games
import utils
import constants

class AteBall():
    def __init__(self, port):
        self.ipc = utils.IPC()

        self.click_offset = [0, 0]

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
                    if not self.processing_play_request.is_set() and self.active_game is None:
                        threading.Thread(target=self.play, args=(msg,), daemon=True).start()
                    else:
                        response = { "type" : "BUSY"}
                elif m_type == "show-realtime":
                    if self.active_game:
                        if self.active_game.show_realtime_event.is_set():
                            self.active_game.show_realtime_event.clear()
                        else:
                            self.active_game.show_realtime_event.set()
                elif m_type == "cancel":
                    self.cancel()
                elif m_type == "quit":
                    self.quit()
                else:
                    pass
            except q.Empty() as e:
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
                    response["msg"] = str(self.exception)

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
        
            location = data['location'] if "location" in data else ""
        
            Game = getattr(games, data['gamemode'].upper())
            self.active_game = Game(self.ipc, location, daemon=True)
            self.active_game.start()
        except Exception as e:
            if isinstance(e, KeyError):
                self.logger.error(f"error playing game: invalid gamemode location - {data['location']}")
            elif isinstance(e, AttributeError):
                self.logger.error(f"error playing game: invalid gamemode - {data['gamemode']}")
            else:
                self.logger.error(f"error playing game: {traceback.format_exc()}")

            self.ipc.send_message({"type" : "GAME-CANCELLED"})
        finally:
            self.processing_play_request.clear()

    def cancel(self):
        if self.active_game:
            self.active_game.cancel()
            self.active_game = None
            self.ipc.send_message({"type" : "GAME-CANCELLED"})

    ###menu

    def quit(self):
        self.ipc.quit()
        self.quit_event.set()