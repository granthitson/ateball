const { contextBridge, ipcRenderer } = require('electron')

contextBridge.exposeInMainWorld('api', {
    test : () => ipcRenderer.invoke('test'),
    inject_css: (type) => ipcRenderer.invoke('get-css', type),
    send_message: (data) => ipcRenderer.invoke('send_message', data),
    ateball : {
        state: {
            login: () => {
                return localStorage.getItem("accessToken");
            },
            // ...ipcRenderer.send("ateball-status")
        },
        start: () => ipcRenderer.send('ateball-start'),
        started: (callback) => ipcRenderer.on('ateball-started', callback),
        stop: () => ipcRenderer.send('ateball-stop'),
        stopped: (callback) => ipcRenderer.on('ateball-stopped', callback),
        log_message: (callback) => ipcRenderer.on('ateball-log', callback)
    },
});