const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('api', {
    test : () => ipcRenderer.invoke('test'),
    inject_css: (type) => ipcRenderer.invoke('get-css', type),
    ateball : {
        state: () => ipcRenderer.invoke('ateball-state'),
        start: () => ipcRenderer.send('ateball-start'),
        on_start: (callback) => ipcRenderer.on('ateball-started', callback),
        on_connected: (callback) => ipcRenderer.on('ateball-connected', callback),
        play: (data) => ipcRenderer.invoke('game-play', data),
        on_play: (callback) => ipcRenderer.on('game-playing', callback),
        game : {
            cancel: () => ipcRenderer.send('game-cancel'),
            on_cancel: (callback) => ipcRenderer.on('game-cancelled', callback),
            on_stop: (callback) => ipcRenderer.on('game-stopped', callback),
        },
        stop: () => ipcRenderer.send('ateball-stop'),
        on_stop: (callback) => ipcRenderer.on('ateball-stopped', callback),
        log_message: (callback) => ipcRenderer.on('ateball-log', callback),
    },
});