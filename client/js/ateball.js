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
					suit: null,
					turn_num: null,
					targets: null,
					balls: {},
					round: {
						started: false,
						ball_clusters: {},
						ball_paths: {},
					}
				}
			}
		};
		this.original_state = structuredClone(this.state);

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

	start_game(msg) {
		this.state.ateball.game.suit = (msg.data.suit) ? null : undefined;
		this.state.ateball.game.balls = msg.data.balls;
		
		this.state.ateball.pending = false;
		this.state.ateball.game.started = true;

		this.toggle_game_controls();
		this.window.webContents.send("game-started");
	}

	select_ball_path(data) {
		this.send_message({ "type" : "select-ball-path", "data" : data});
	}

	update_targets(data) {
		this.state.ateball.game.targets = data.targets; 
		this.send_message({ "type" : "update-targets", "data" : data});
	}

	realtime_configure(data) {
		this.send_message({ "type" : "realtime-configure", "data" : data});
	}

	cancel_game() {
		this.send_message({ "type" : "cancel" });
	}

	stop_game() {
		this.send_message({ "type" : "stop" });
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

		this.process = subpy.spawn(process.env.PYTHON, ["./backend/main.py"], { stdio: 'pipe', windowsHide: false });

		this.process.once('spawn', function () {
			self.send_message({ "type" : "init" });

			console.log("ateball process spawned");
			start_resolve();

			self.process.stdout.removeAllListeners('data');
			self.process.stdout.on('data', function (data) {
				self.process_message(data);
			});

			self.process.stderr.on('data', function(data) {
				self.process_message(data);
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
			console.error('ateball process failed to start', err);
			start_reject();
		});

		return this.started;
	};
	
	process_message(data) {
		var messages = data.toString().trim().split(/\r?\n|\r/g);
		messages.forEach((msg) => {
			try {
				var p_msg = JSON.parse(msg);
				if (p_msg.id) {
					this.pending[id]()
				} else {
					switch (p_msg.type) {
						case "BUSY":
							console.log("ateball busy");
							break;
						case "INIT":
							console.log("ateball initialized");
							this.state.process.connected = true;
							break;
						case "GAME-START":
							console.log("game started");
							this.start_game(p_msg);
							break;
						case "ROUND-START":
							console.log("round started");
							this.state.ateball.game.turn_num = p_msg.data.turn_num;
							this.state.ateball.game.round.started = true;
							this.state.ateball.game.round.ball_clusters = {};
							this.state.ateball.game.round.ball_paths = {};

							break;
						case "UPDATE-BALL-STATE":
							this.state.ateball.game.balls = p_msg.data.balls;
							break;
						case "SUIT-SELECT":
							this.state.ateball.game.suit = p_msg.data.suit;
							break;
						case "REALTIME-STREAM":
							this.state.ateball.game.balls = p_msg.data.balls;

							if (p_msg.data.image) {
								this.window.webContents.send("realtime-stream", { data: p_msg.data.image});
							}
							break;
						case "ROUND-UPDATE":
							if (p_msg.data) {
								switch (p_msg.data.type) {
									case "SET-BALL-CLUSTERS":
										this.state.ateball.game.round.ball_clusters = p_msg.data.ball_clusters;
										break;
									case "SET-BALL-PATHS":
										this.state.ateball.game.round.ball_paths = p_msg.data.ball_paths;
										break;
									default:
										break;
								}
							}

							break;
						case "ROUND-END":
						case "ROUND-COMPLETE":
							console.log("round ended");
							this.reset_round_state();

							break;
						case "GAME-EXCEPTION":
						case "GAME-CANCELLED":
						case "GAME-STOPPED":
							console.log("game stopped");
							this.end_game();
							break;
						case "GAME-END":
							console.log("game ended");
							this.end_game();
							break;
						default:
							console.log("unrecognized message: ", p_msg);
							break;
					}
				}
			} catch (e) {
				this.window.webContents.send("ateball-log", msg);
			}
		});
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

	reset_round_state() {
		this.state.ateball.game.round = structuredClone(this.original_state.ateball.game.round);
	}

	reset_game_state() {
		this.state.ateball = structuredClone(this.original_state.ateball);
	}

	reset_ateball_state() {
		this.state = structuredClone(this.original_state);
	}

	kill() {
		if (this.process != null) {
			this.process.stdout.removeAllListeners('data');
			this.process.kill();
		}
	}
}

module.exports = { Ateball: Ateball };