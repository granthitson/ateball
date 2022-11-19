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

app.on('ready', async () => {
	// Create the browser window.
	let factor = screen.getPrimaryDisplay().scaleFactor;
	const window = new BrowserWindow({
		icon: "",
		width: 1200, // / factor,
		height: 600, // / factor,
		autoHideMenuBar: !app.isPackaged,
		alwaysOnTop: true,
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

	ipcMain.handle('get-css', async (e, type) => {
		console.log("injecting css: ", type);
		return new Promise((resolve, reject) => {
			fs.readFile(path.join(__dirname, 'ateball-js/css/' + type + '.css'), 'utf8', (err, data) => {
				if (err) reject();
				resolve(data);
			});
		})
	});

	ipcMain.on('ateball-start', async (e) => {
		console.log("starting ateball");
		if (!ateball == null) {
			(new Promise((resolve, reject) => {
				spawn_ateball(window);
				resolve();
			})).catch((e) => {
				console.log("could not start ateball", e);
			});
		}
	});

	ipcMain.on('ateball-status', async (e) => {
		return (new Promise((resolve, reject) => {
			if (ateball != null) {
				console.log("getting ateball status");
				ateball.send({ action : "status", callback : (data) => {
					resolve(data); // figure out way to return from ateball
				}});
			} else {
				resolve(null);
			}
		})).catch((e) => {
			console.log("could not get ateball status", e);
		});
	});

	ipcMain.on('ateball-stop', async (e) => {
		console.log("stopping ateball");
		(new Promise((resolve, reject) => {
			try {
				ateball.send({ action : "quit", callback : (data) => {
					ateball.kill();
					resolve();
				}});
			} catch (e) {
				reject();
			}
		})).catch((e) => {
			console.log("could not stop ateball");
		});
	});

	var ateball = null;
	window.webContents.once('dom-ready', () => {
		spawn_ateball(window);
	});

	ipcMain.handle("test", (e) => {
		console.log("test");
	});
});

var ateball = undefined;
const spawn_ateball = (window) => {
	ateball = subpy.spawn( "python", [ "./ateball-py/main.py", process.env.PORT], { windowsHide: false});

	ateball.on('spawn', function () {
		console.log("ateball process spawned");
		window.webContents.send("ateball-started");
	});

	ateball.stdout.on('data', function (data) {
		var data = data.toString().trim().split(/\r?\n|\r/g);
		data.forEach((msg) => {
			window.webContents.send("ateball-log", msg);
		})
	});

	// ateball.stdout.on('end', function (data) {
	// 	console.log("ateball python done yo");
	// });

	// ------------------

	ipcMain.once('send-message', async (e, msg) => {
		console.log("sending msg: ", msg);
		return new Promise((resolve, reject) => {
			// send message
			resolve();
		})
	});

	// ===================

	ateball.on('exit', function() {
		console.log("ateball process exited");
		window.webContents.send("ateball-stopped");
	});

	process.on('SIGTERM', () => {
		if (ateball.pid) ateball.kill();
	});
	
	process.on('SIGINT', () => {
		if (ateball.pid) ateball.kill();
	});
}

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