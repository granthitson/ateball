const { contextBridge, ipcRenderer } = require('electron')

contextBridge.exposeInMainWorld('api', {
    test : () => ipcRenderer.invoke('test'),
    inject_css: (type) => ipcRenderer.invoke('get-css', type),
    send_message: (data) => ipcRenderer.invoke('send_message', data),
    ateball : {
        state: () => ipcRenderer.invoke('ateball-state'),
        start: () => ipcRenderer.send('ateball-start'),
        on_start: (callback) => ipcRenderer.on('ateball-started', callback),
        on_connected: (callback) => ipcRenderer.on('ateball-connected', callback),
        stop: () => ipcRenderer.send('ateball-stop'),
        on_stop: (callback) => ipcRenderer.on('ateball-stopped', callback),
        log_message: (callback) => ipcRenderer.on('ateball-log', callback),
    },
});