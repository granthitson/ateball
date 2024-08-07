const electron = require('electron');
const { app, BrowserWindow, session, globalShortcut, ipcMain } = electron;
const windowStateKeeper = require('electron-window-state');

const dotenv = require('dotenv');
const debounce = require('debounce');
const path = require('path');
const URL = require('url').URL

const { Webview } = require('./client/js/webview');
const { Ateball } = require('./client/js/ateball');

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
	let window_state = windowStateKeeper();

	// Create the browser window.
	window = new BrowserWindow({
		icon: "",
		x: window_state.x,
		y: window_state.y,
		width: 1200,
		height: 956,
		minHeight: 956,
		menuBarVisible: !app.isPackaged,
		autoHideMenuBar: !app.isPackaged,
		resizable: false,
		useContentSize: true,
		fullscreenable: false,
		kiosk: true,
		webPreferences: {
			title: process.env.APP_NAME,
			devTools: !app.isPackaged,
			sandbox: true,
			webviewTag: true,
			preload: path.join(__dirname, 'client/js/preload.js'),
		}
	});
	
	window_state.manage(window);
	window.on('resize', debounce(window_state.saveState, 500))
	window.on('move', debounce(window_state.saveState, 500))

	window.webContents.openDevTools();

	// clear persistent data
	session.defaultSession.clearStorageData();
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

	var webview = new Webview(window);
	var ateball = new Ateball(window);
	
	window.loadFile(path.join(__dirname, "client/html/index.html"));
	window.webContents.on('did-finish-load', (e) => {
		window.focus();
	});

	window.webContents.on('dom-ready', (e) => {
		if (ateball && ateball.state.ateball.game.started) {
			ateball.stop_game()
		}
	});

	window.webContents.once('dom-ready', (e) => {
		ateball.start();
	});

	window.on('closed', (e) => {
		try {
			window_state.saveState(window);
			ateball.kill("SIGINT");
		} catch (e) {
			console.log("couldnt close ateball");
		}
	});

	ipcMain.handle('get-state', (e) => {
		return {
			webview : webview.get_state(),
			...ateball.get_state()
		}
	});

	// --- webview events ---

	ipcMain.handle('webview-format', (e, type) => {
		return webview.format(type);
	});

	ipcMain.on('webview-formatted', (e) => {
		console.log("formatting complete");
		webview.state.formatted = true;
	});

	ipcMain.on('webview-loaded', (e, location, logged_in) => {
		webview.state.loaded = true;

		webview.state.menu = (location || null);
		webview.state.logged_in = (logged_in !== null);
	});

	// --- ateball events ---

	ipcMain.on('ateball-start', (e) => {
		ateball.start();
	});

	ipcMain.on('game-play', (e, data) => {
		ateball.play_game(data);
	});

	ipcMain.on('select-ball-path', (e, data) => {
		ateball.select_ball_path(data);
	});

	ipcMain.on('update-targets', (e, data) => {
		ateball.update_targets(data);
	});

	ipcMain.on('executed-path', (e) => {
		ateball.executed_path();
	});

	ipcMain.on('realtime-round-decrement', (e) => {
		ateball.realtime_set_current_round(-1);
	});

	ipcMain.on('realtime-round-increment', (e) => {
		ateball.realtime_set_current_round(1);
	});

	ipcMain.on('realtime-configure', (e, data) => {
		ateball.realtime_configure(data.type, data.value);
	});
	
	ipcMain.on('game-cancel', (e) => {
		ateball.cancel_game();
	});

	ipcMain.on('game-stop', (e) => {
		ateball.stop_game();
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
	window_state.saveState(window);
	ateball.kill();
});

process.on('SIGINT', () => {
	window_state.saveState(window);
	ateball.kill();
});