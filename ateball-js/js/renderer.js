var webview = document.querySelector("webview");

webview.addEventListener('dom-ready', () => {
    // webview.openDevTools();
});

webview.addEventListener("did-navigate", () => {
    window.api.inject_css("webview").then((css) => {
        webview.insertCSS(css);
    }).then(() => {
        // don't really like this but works fairly reliably
        setTimeout(() => {
            toggleGUIElements(get_state());
        }, 1000);
    });
});

var log = document.querySelector('#debug_console_2');
['log','debug','info','warn','error'].forEach(function (verb) {
    console[verb] = (function (method, verb, log) {
        return function () {
            method.apply(console, arguments);
            var msg = document.createElement('span');
            msg.classList.add("log-object");
            msg.innerHTML += Array.prototype.slice.call(arguments).join(' ');
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
    log_message("<div class='empty'></div>");
});

window.api.ateball.log_message((e, msg) => {
    log_message(msg);
});

const debug = document.querySelector("#debug_console_1");
const log_message = (msg) => {
    var log = document.createElement('span');
    log.classList.add("log-object");
    log.innerHTML += msg;
    debug.prepend(log);
}

const get_state = async () => {
    var state = {};
    
    await window.api.ateball.state().then((a_state) => { state.ateball = a_state; });
    await webview.executeJavaScript('location.pathname || null').then((m_state) => { state.menu = m_state; });
    await webview.executeJavaScript('(localStorage.getItem("accessToken") !== null) ? true : false').then((l_state) => { state.logged_in = l_state; });

    return state;
}