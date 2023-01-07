var webview = document.querySelector("webview");

webview.addEventListener('dom-ready', () => {
    webview.openDevTools();
});

webview.addEventListener("did-navigate", () => {
    webview.style.display = "none";
    window.api.inject_css("webview").then((css) => {
        webview.insertCSS(css);
    });
});

webview.addEventListener('ipc-message', (e) => {
    console.log(e.channel)
    switch (e.channel) {
        case "formatting-complete":
            webview.style.display = "flex";
            toggleGUIElements(get_state(false));
            break;
        case "loaded":
            console.log("loaded");
            toggleGUIElements(get_state(true));
            break;
    }
  })

var log = document.querySelector('#debug_console_2');
['log','debug','info','warn','error'].forEach(function (verb) {
    console[verb] = (function (method, verb, log) {
        return function () {
            method.apply(console, arguments);
            var msg = document.createElement('span');
            msg.classList.add("log-object");
            msg.innerHTML += Array.prototype.slice.call(arguments).map(x => { return (typeof(x) === 'object') ? JSON.stringify(x) : x } ).join(' ');
            log.prepend(msg);
        };
    })(console[verb], verb, log);
});

webview.addEventListener('console-message', (e) => {
    console.log(e.message);
});

// ----------------------

var start = document.querySelector("#ateball-start");
start.addEventListener("click", (e) => {
    window.api.ateball.start();
});

window.api.ateball.on_start( (e) => {
    console.log("Ateball started");
    toggleAteballControls(true);
});

window.api.ateball.on_connected( (e) => {
    console.log("Ateball connected");
    toggleGUIElements(get_state());
});

var stop = document.querySelector("#ateball-stop");
stop.addEventListener("click", (e) => {
    window.api.ateball.stop();
});

window.api.ateball.on_stop(() => {
    console.log("Ateball stopped");
    toggleGUIElements(Promise.resolve({}));
    toggleAteballControls(false);
    
    var empty = document.createElement('div')
    empty.classList.add('empty')
    log_message(empty);
});

window.api.ateball.log_message((e, msg) => {
    log_message(msg);
});

const debug = document.querySelector("#debug_console_1");
const log_message = (msg) => {
    var log = document.createElement('span');
    log.classList.add("log-object");

    if (typeof(msg) === "object") {
        log.appendChild(msg);
    } else if (typeof(msg) === "string") {
        log.innerText += msg;
    }

    debug.prepend(log);
}

const state = { loaded: false };
const get_state = async (loaded=null) => {
    if (loaded != null) {
        state.loaded = loaded;
    }

    await window.api.ateball.state().then((a_state) => { state.ateball = a_state; });
    await webview.executeJavaScript('location.pathname || null').then((m_state) => { state.menu = m_state; });
    await webview.executeJavaScript('(localStorage.getItem("accessToken") !== null) ? true : false').then((l_state) => { state.logged_in = l_state; });

    return state;
}