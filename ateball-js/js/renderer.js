var webview = document.querySelector("webview");

webview.addEventListener('dom-ready', () => {
    // webview.openDevTools();
});

webview.addEventListener("did-navigate", () => {
    console.log("navigate", webview.getURL());
    window.api.inject_css("webview").then((css) => {
        webview.insertCSS(css);
    });
    webview.send("format");
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

window.api.ateball.started(() => {
    console.log("Ateball started");
    // allowGUIInteraction(true);
    toggleAteballControls(true);
    toggleGUIElements(window.api.ateball.state);
});

window.api.ateball.stopped(() => {
    console.log("Ateball stopped");
    allowGUIInteraction(false);
    toggleAteballControls(false);
    log_message("<div class='empty'></div>");
});

window.api.ateball.log_message((e, msg) => {
    log_message(msg);
});

const log_message = (msg) => {
    var debug = document.querySelector("#debug_console_1");
    var log = document.createElement('span');
    log.classList.add("log-object");
    log.innerHTML += msg;
    debug.prepend(log);
}