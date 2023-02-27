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
		var self = this;
		this.state.ateball.pending = true;

		this.window.setAlwaysOnTop(true, 'pop-up-menu');
		self.send_message(msg, (resp) => {
			console.log(resp);
			return resp;
		});
	}

	cancel_game() {
		this.window.setAlwaysOnTop(false);
		this.send_message({ "type" : "cancel" });
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

					self.reset_state();
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
		console.log("processing message: ", msg);
		if (msg.id) {
			this.pending[id]()
		} else {
			switch (msg.type) {
				case "BUSY":
					this.window.webContents.send("ateball-busy");
					break;
				case "INIT":
					this.state.process.connected = true;
					this.window.webContents.send("ateball-connected");
					break;
				case "GAME-START":
					this.state.ateball.pending = false;
					this.state.ateball.game.started = true;
					this.window.webContents.send("game-playing");
					break;
				case "ROUND-START":
					this.state.ateball.game.round.started = true;
					this.window.webContents.send("game-round-start");
					break;
				case "ROUND-END":
					this.state.ateball.game.round.started = false;
					this.window.webContents.send("game-round-end");
					break;
				case "GAME-CANCELLED":
					console.log("cancelling");
					this.state.ateball.pending = false;
					this.state.ateball.game.started = false;
					this.state.ateball.game.round.started = false;
					this.window.webContents.send("game-cancelled");
					break;
				case "GAME-END":
					console.log("ending game");
					this.state.ateball.pending = false;
					this.state.ateball.game.started = false;
					this.state.ateball.game.round.started = false;
					this.window.webContents.send("game-stopped");
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

	get_state() {
		return this.state;
	}

	reset_state() {
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