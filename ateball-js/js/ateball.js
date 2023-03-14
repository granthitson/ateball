const subpy = require( "child_process" );

class Ateball {
	constructor(window) {
		this.window = window;
		
		this.process = null;

		this.state = {
			process : {
				started: false,
				connected: false,
			},
			ateball : {
				pending: false,
				game: {
					started: false,
					round: {
						started: false
					}
				}
			}
		};

		this.started = null;
		this.stopped = null;

		this.pending = {};
	}

	start() {
		if (this.process == null) {
			console.log("starting ateball");
			this.spawn().then(() => {
				this.state.process.started = true;
				this.window.webContents.send("ateball-started");
			}).catch((e) => {
				this.state.process.started = false;
				console.log("could not start ateball", e);
			});
		}
	}

	play_game(msg) {
		this.state.ateball.pending = true;

		this.window.setAlwaysOnTop(true, 'pop-up-menu');
		this.send_message(msg);
	}

	realtime_configure(data) {
		this.send_message({ "type" : "realtime-configure", "data" : data});
	}

	cancel_game() {
		this.window.setAlwaysOnTop(false);
		this.send_message({ "type" : "cancel" });
	}

	end_game() {
		this.window.setAlwaysOnTop(false);
		this.reset_game_state();
		this.toggle_game_controls();

		this.window.webContents.send("game-ended");
	}

	stop() {
		if (this.process != null) {
			console.log("stopping ateball");
			this.send_message({ "type" : "quit" });
			var timeout = setTimeout(() => {
				if (this.process) {
					this.kill();
				}
			}, 10000);

			this.stopped.then(() => {
				clearTimeout(timeout);
			});
		}
	}

	spawn() {
		var self = this;

		var start_resolve, start_reject;
		this.started = new Promise((resolve, reject) => {
			start_resolve = resolve;
			start_reject = reject;
		}); 

		var stop_resolve, stop_reject;
		this.stopped = new Promise((resolve, reject) => {
			stop_resolve = resolve;
			stop_reject = reject;
		});

		this.process = subpy.spawn("python", ["./ateball-py/main.py"], { stdio: 'pipe', windowsHide: false });

		this.process.once('spawn', function () {
			console.log("ateball process spawned");
			start_resolve();

			self.process.stdout.removeAllListeners('data');
			self.process.stdout.on('data', function (data) {
				var data = data.toString().trim().split(/\r?\n|\r/g);
				data.forEach((msg) => {
					try {
						var p_msg = JSON.parse(msg);
						self.process_message(p_msg);
					} catch (e) {
						self.window.webContents.send("ateball-log", msg);
					}
				});
			});
	
			self.process.once('exit', function () {
				console.log("ateball process exited");
				stop_resolve();
				if (!self.window.isDestroyed()) {
					self.window.setAlwaysOnTop(false);

					self.reset_ateball_state();
					self.toggle_game_controls();
					self.process = null;
					self.window.webContents.send("ateball-stopped");
				}
			});
		});

		this.process.once('error', (err) => {
			console.error('ateball process failed to start');
			start_reject();
		});

		return this.started;
	};
	
	process_message(msg) {
		if (msg.id) {
			this.pending[id]()
		} else {
			switch (msg.type) {
				case "BUSY":
					console.log("ateball busy");
					break;
				case "INIT":
					console.log("ateball initialized");
					this.state.process.connected = true;
					break;
				case "GAME-START":
					console.log("game started");
					this.state.ateball.pending = false;
					this.state.ateball.game.started = true;

					this.toggle_game_controls();
					break;
				case "ROUND-START":
					console.log("round started");
					this.state.ateball.game.round.started = true;
					break;
				case "REALTIME-STREAM":
					this.window.webContents.send("realtime-stream", { data: msg.data});
					break;
				case "ROUND-END":
					console.log("round ended");
					this.state.ateball.game.round.started = false;
					break;
				case "GAME-EXCEPTION":
				case "GAME-CANCELLED":
					console.log("game cancelled");
					this.end_game();
					break;
				case "GAME-END":
					console.log("game ended");
					this.end_game();
					break;
				default:
					console.log("unrecognized message type: ", msg.type);
					break;
			}
		}
	}

	send_message(msg, callback = null) {
		try {
			var id = (new Date()).getTime().toString(36);
			if (callback) {
				this.pending[id] = callback;
			}

			msg = { 
				id : id,
				...msg
			}

			console.log("sending msg : ", msg);
			
			this.process.stdin.write(JSON.stringify(msg) + "\n");
		} catch (e) {
			console.error("could not send message");
		}
	}

	toggle_game_controls() {
		var bounds = this.window.getContentBounds();

		if (this.state.ateball.game.started) {
			bounds.height = 956;
		} else {
			bounds.height = 600;
		}
		
		this.window.setContentBounds(bounds);
	}

	get_state() {
		return this.state;
	}

	reset_game_state() {
		this.state.ateball.pending = false;
		this.state.ateball.game.started = false;
		this.state.ateball.game.round.started = false;
	}

	reset_ateball_state() {
		this.state = {
			process : {
				started: false,
				connected: false,
			},
			ateball : {
				pending: false,
				game: {
					started: false,
					round: {
						started: false
					}
				}
			}
		};
	}

	kill() {
		if (this.process != null) {
			this.process.stdout.removeAllListeners('data');
			this.process.kill();
		}
	}
}

module.exports = { Ateball: Ateball };