const electron = require('electron');
const URL = require('url').URL
const dotenv = require('dotenv');
const subpy = require( "child_process" );
const fs = require('fs');

const { app, BrowserWindow, session, screen, globalShortcut, ipcMain } = electron;
const path = require('path');

dotenv.config();

// prevent flashing when loading page
app.allowRendererProcessReuse = true;
app.commandLine.appendSwitch("disable-site-isolation-trials");

var ateball = null;

app.on('ready', async () => {
	// Create the browser window.
	let factor = screen.getPrimaryDisplay().scaleFactor;
	const window = new BrowserWindow({
		icon: "",
		width: 1200, // / factor,
		height: 600, // / factor,
		autoHideMenuBar: !app.isPackaged,
		// alwaysOnTop: true,
		minimizable: false,
		resizable: false,
		useContentSize: true,
		fullscreenable: false,
		kiosk: true,
		webPreferences: {
			title: "AteballBot",
			devTools: !app.isPackaged,
			sandbox: true,
			webSecurity: false,
			webviewTag: true,
			preload: path.join(__dirname, '/ateball-js/js/preload.js'),
		}
	});
	window.setMenuBarVisibility(false);
	window.webContents.openDevTools();

	// clear persistent data
	session.defaultSession.clearStorageData();

	ateball = new Ateball(window);
	
	window.loadFile(path.join(__dirname, "/ateball-js/html/index.html"));
	window.webContents.on('did-finish-load', () => {
		window.focus();
	});

	window.on('closed', () => {
		try {
			ateball.kill("SIGINT");
		} catch (e) {
			console.log("couldnt close ateball");
		}
	});

	// need to change filter list to inverse of what it is
	session.defaultSession.webRequest.onBeforeRequest({ urls: [process.env.ACCEPTED_FULL_FILTER_LIST] }, (details, callback) => {
		var parsed_url = new URL(details.url);
		console.log(parsed_url);
		if (!process.env.ACCEPTED_ORIGINS_FULL.split(" ").includes(parsed_url)) {
			callback({ cancel: true });
		} else {
			callback({});
		}
	});

	ipcMain.handle('get-css', (e, type) => {
		console.log("injecting css: ", type);
		return new Promise((resolve, reject) => {
			fs.readFile(path.join(__dirname, 'ateball-js/css/' + type + '.css'), 'utf8', (err, data) => {
				if (err) reject();
				resolve(data);
			});
		})
	});

	window.webContents.once('dom-ready', () => {
		ateball.start();
	});

	ipcMain.on('ateball-start', () => {
		ateball.start();
	});

	ipcMain.handle('ateball-state', (e) => {
		return (new Promise((resolve, reject) => {
			resolve(ateball.get_state());
		}));
	});

	ipcMain.on('ateball-status', (e) => {
		return (new Promise((resolve, reject) => {
			if (ateball.process != null) {
				console.log("getting ateball status");
				ateball.get_status().then((data) => {
					resolve(data); // figure out way to return from ateball
				});
			} else {
				resolve(null);
			}
		})).catch((e) => {
			console.log("could not get ateball status", e);
		});
	});

	ipcMain.on('ateball-stop', (e) => {
		ateball.stop();
	});
});

app.on('web-contents-created', (e, contents) => {
	contents.on('will-navigate', (e, url) => {
		var accepted_origins = (app.isPackaged) ? process.env.ACCEPTED_ORIGINS_GUEST.split(" ") : process.env.ACCEPTED_ORIGINS_FULL.split(" ");
		const parsed_url = new URL(url)

		// console.log("parsed url: ", parsed_url.href);
		if (!(accepted_origins.includes(parsed_url.origin))) {
			e.preventDefault();
		}
	});
});

app.on('browser-window-focus', function () {
	globalShortcut.register("CommandOrControl+R", () => {
		console.log("CommandOrControl+R is pressed: Shortcut Disabled");
	});
	globalShortcut.register("CommandOrControl+Shift+R", () => {
		console.log("CommandOrControl+Shift+R is pressed: Shortcut Disabled");
	});
	globalShortcut.register("CommandOrControl+-", () => {
		console.log("Zoom out is pressed: Shortcut Disabled");
	});
	globalShortcut.register("CommandOrControl+=", () => {
		console.log("Zoom in is pressed: Shortcut Disabled");
	});
	globalShortcut.register("F5", () => {
		console.log("F5 is pressed: Shortcut Disabled");
	});
});

app.on('browser-window-blur', function () {
	globalShortcut.unregister('CommandOrControl+R');
	globalShortcut.unregister('CommandOrControl+Shift+R');
	globalShortcut.unregister('CommandOrControl+_');
	globalShortcut.unregister('CommandOrControl+=');
	globalShortcut.unregister('F5');
});

app.on('window-all-closed', () => {
  	app.quit()
});

class Ateball {
	constructor(window) {
		this.window = window;
		
		this.process = null;

		this.state = {
			started: false,
			connected: false,
		};

		this.started = null;
		this.stopped = null;

		this.pending = {};
	}

	start() {
		if (this.process == null) {
			console.log("starting ateball");
			this.spawn().then(() => {
				this.state.started = true;
				this.window.webContents.send("ateball-started");
			}).catch((e) => {
				this.state.started = false;
				console.log("could not start ateball", e);
			});
		}
	}

	stop() {
		if (this.process != null) {
			console.log("stopping ateball");
			this.send_message({ action : "quit" });
			this.stopped.catch((e) => {
				console.log("could not stop ateball", e);
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
					self.state.started = false;
					self.state.connected = false;
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
		switch (msg.action) {
			case "STATUS":
				if (msg.type == "INIT") {
					this.state.connected = true;
					this.window.webContents.send("ateball-connected");
				}
				break;
		}
	}

	send_message(msg, callback = null) {
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
	}

	get_state() {
		return this.state;
	}

	get_status() {
		var status = new Promise((resolve, reject) => {
			this.send_message({ action : "status" });
		});
		this.pending[id] = status;

		return status;
	}

	kill() {
		if (this.process != null) {
			this.process.stdout.removeAllListeners('data');
			this.process.kill();
		}
	}
}

process.on('SIGTERM', () => {
	ateball.kill();
});

process.on('SIGINT', () => {
	ateball.kill();
});