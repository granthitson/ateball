const electron = require('electron');
const { app, BrowserWindow, session, screen, globalShortcut, ipcMain, webContents } = electron;

const dotenv = require('dotenv');
const path = require('path');
const URL = require('url').URL
const fs = require('fs');

const { Ateball } = require('./ateball-js/js/ateball');

dotenv.config();

// prevent flashing when loading page
app.allowRendererProcessReuse = true;
app.commandLine.appendSwitch("disable-site-isolation-trials");

var window = null;

const instance_lock = app.requestSingleInstanceLock();
if (instance_lock) {
	app.on('second-instance', (event, argv, cwd) => {
		if (window) {
			if (window.isMinimized()) window.restore();
			window.focus();
		}
	});
} else {
	app.quit();
}


app.on('ready', async () => {
	// Create the browser window.
	let factor = screen.getPrimaryDisplay().scaleFactor;
	window = new BrowserWindow({
		icon: "",
		width: 1200, // / factor,
		height: 600, // / factor,
		autoHideMenuBar: !app.isPackaged,
		minimizable: false,
		resizable: false,
		useContentSize: true,
		fullscreenable: false,
		kiosk: true,
		webPreferences: {
			title: process.env.APP_NAME,
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

	var ateball = new Ateball(window);
	
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

	session.defaultSession.webRequest.onBeforeRequest({ urls: process.env.ACCEPTED_FULL_FILTER_LIST.split(" ") }, (details, callback) => {
		var parsed_url = new URL(details.url);
		if (parsed_url.protocol != "file:") {
			if (!process.env.ACCEPTED_HOST_FULL.split(" ").includes(parsed_url.host)) {
				callback({ cancel: true });
			} else {
				callback({});
			}
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
		return ateball.get_state();
	});

	ipcMain.handle('game-play', (e, data) => {
		return (new Promise((resolve, reject) => {
			if (ateball.process != null) {
				ateball.play_game({ type: "play", ...data});
				resolve(true);
			} else {
				resolve(null);
			}
		})).catch((e) => {
			console.log("could not get play game", e);
		});
		
	});

	ipcMain.on('game-cancel', (e) => {
		ateball.cancel_game();
	});

	ipcMain.on('ateball-stop', (e) => {
		ateball.stop();
	});
});

app.on('web-contents-created', (e, contents) => {
	contents.on('will-navigate', (e, url) => {
		var accepted_hosts = (app.isPackaged) ? process.env.ACCEPTED_HOST_GUEST.split(" ") : process.env.ACCEPTED_HOST_FULL.split(" ");
		const parsed_url = new URL(url)

		if (!(accepted_hosts.includes(parsed_url.host))) {
			console.log("blocked url: ", parsed_url.href);
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

process.on('SIGTERM', () => {
	ateball.kill();
});

process.on('SIGINT', () => {
	ateball.kill();
});