const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('api', {
    get_state : () => ipcRenderer.invoke('get-state'),
    webview : {
        format: (type) => ipcRenderer.invoke('webview-format', type)
    },
    ateball : {
        state: () => ipcRenderer.invoke('ateball-state'),
        start: () => ipcRenderer.send('ateball-start'),
        on_start: (callback) => ipcRenderer.on('ateball-started', callback),
        play: (data) => ipcRenderer.send('game-play', data),
        game : {
            realtime : {
                on_stream: (callback) => ipcRenderer.on('realtime-stream', callback)
            },
            cancel: () => ipcRenderer.send('game-cancel'),
            on_end: (callback) => ipcRenderer.on('game-ended', callback),
        },
        stop: () => ipcRenderer.send('ateball-stop'),
        on_stop: (callback) => ipcRenderer.on('ateball-stopped', callback),
        log_message: (callback) => ipcRenderer.on('ateball-log', callback),
    },
});