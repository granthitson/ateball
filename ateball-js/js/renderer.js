var webview = document.querySelector("webview");

webview.addEventListener('dom-ready', () => {
    webview.openDevTools();
});

webview.addEventListener('crashed', (e, args) => {
    console.log("crashed", e, args);
    webview.reload();
});

webview.addEventListener('destroyed', (e, args) => {
    console.log("destroyed", e, args);
    webview.reload();
});

webview.addEventListener("did-navigate", () => {
    webview.style.display = "none";
    window.api.inject_css("webview").then((css) => {
        webview.insertCSS(css);
    });
});

webview.addEventListener('ipc-message', (e) => {
    switch (e.channel) {
        case "formatting-complete":
            webview.style.display = "flex";
            toggleGUIElements(get_state(false));
            break;
        case "loaded":
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
            msg.classList.add("log-object", verb);
            msg.innerHTML += Array.prototype.slice.call(arguments).map(x => { return (typeof(x) === 'object') ? JSON.stringify(x) : x } ).join(' ');
            log.prepend(msg);
        };
    })(console[verb], verb, log);
});

const log_level = {0 : "info", 1 : "log", 2 : "warn", 3 : "error"};
webview.addEventListener('console-message', (e) => {
    console[log_level[e.level]](e.message);
});

// ----------------------

window.api.ateball.on_start( (e) => {
    console.log("Ateball started");
    toggleAteballControls(true);
});

window.api.ateball.on_connected( (e) => {
    console.log("Ateball connected");
    toggleGUIElements(get_state());
});

window.api.ateball.on_play( (e) => {
    console.log("Ateball playing");
    toggleGUIElements(get_state());
});

window.api.ateball.on_cancel( (e) => {
    console.log("Ateball cancelled");
    toggleGUIElements(get_state());
});

window.api.ateball.on_stop(() => {
    console.log("Ateball stopped");
    toggleGUIElements(get_state());
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
        log.innerText += truncate(msg);
    }

    debug.prepend(log);
}

const truncate = (msg) => {
    return (msg.length > 255) ? msg.substring(0, 255) + "..." : msg;
}

const state = { loaded: false };
const get_state = async (loaded=null) => {
    if (loaded != null) {
        state.loaded = loaded;
    }

    await window.api.ateball.state().then((a_state) => { Object.assign(state, a_state); });
    await webview.executeJavaScript('location.pathname || null').then((m_state) => { state.menu = m_state; });
    await webview.executeJavaScript('(localStorage.getItem("accessToken") !== null) ? true : false').then((l_state) => { state.logged_in = l_state; });

    return state;
}