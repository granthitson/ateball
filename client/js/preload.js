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
            on_start: (callback) => ipcRenderer.on('game-started', callback),
            round : {
                select_ball_path: (data) => ipcRenderer.send('select-ball-path', data),
                on_target_path: (callback) => ipcRenderer.on('target-path', callback),
                on_execute_path: (callback) => ipcRenderer.on('execute-path', callback)
            },
            update_targets: (data) => ipcRenderer.send('update-targets', data),
            realtime : {
                configure: (data) => ipcRenderer.send('realtime-configure', data),
                on_stream: (callback) => ipcRenderer.on('realtime-stream', callback)
            },
            cancel: () => ipcRenderer.send('game-cancel'),
            stop: () => ipcRenderer.send('game-stop'),
            on_end: (callback) => ipcRenderer.on('game-ended', callback),
        },
        stop: () => ipcRenderer.send('ateball-stop'),
        on_stop: (callback) => ipcRenderer.on('ateball-stopped', callback),
        log_message: (callback) => ipcRenderer.on('ateball-log', callback),
    },
});