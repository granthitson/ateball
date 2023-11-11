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
    window.api.webview.format("webview").then((css) => {
        webview.insertCSS(css);
    });
});

webview.addEventListener("ipc-message", (e) => {
    switch (e.channel) {
        case "mousemove":
        case "mouseup":
        case "mousedown":
            let data = e.args[0];
            ateball_mouse.current_x = data.x;
            ateball_mouse.current_y = data.y;
            break;
        default:
            console.log(`unknown channel: ${e.channel}`);
            break;
    }
});

window.api.ateball.on_start( (e) => {
    console.log("Ateball started");
    toggleButtonSpinner(start, false);
    debug.innerHTML = "";
});

window.api.ateball.game.on_start(() => {
    console.log("game started");
    closeNavigationMenus(gamemode_controls);
});

var image_stream = Promise.resolve();
window.api.ateball.game.realtime.on_stream( (e, msg) => {
    image_stream = new Promise((resolve, reject) => {
        var realtime = document.querySelector("#realtime canvas");
        var context = realtime.getContext("2d");

        var image = new Image();
        image.onload = function() {
            context.drawImage(image, 0, 0);
            resolve();
        };
        image.onerror = function() {
            reject();
        }
        image.src = msg.data;
    });
});

const ateball_mouse = new AteballMouse(webview);

window.api.ateball.game.round.on_execute_path((e, data) => {
    ateball_mouse.execute_path(data.path);
});

window.api.ateball.game.round.on_target_path((e, data) => {
    // need to wait for menu items to be added 
    waitForElement(`.ball_path[data-id="${data.id}"]`, ball_path_menu).then(() => {
        try {
            var ball_path = document.querySelector(`.ball_path[data-id="${data.id}"]`);
            ball_path.click();
        } catch (e) {
            console.error("could not target path: ", e);
        }   
    });
});

window.api.ateball.game.on_end((e) => {
    console.log("game ended");
    toggleButtonSpinner(game_stop, false);
    closeNavigationMenus(game_controls);
    ateball_mouse.cancel();

    image_stream.then(() => {
        var realtime = document.querySelector("#realtime canvas");
        var context = realtime.getContext("2d");
        context.clearRect(0, 0, realtime.width, realtime.height);
    });
});

window.api.ateball.on_stop(() => {
    console.log("Ateball stopped");
    toggleButtonSpinner(game_stop, false);
    toggleButtonSpinner(stop, false);
    ateball_mouse.cancel();

    var realtime = document.querySelector("#realtime canvas");
    var context = realtime.getContext("2d");
    context.clearRect(0, 0, realtime.width, realtime.height);

    var empty = document.createElement('div')
    empty.classList.add('empty')
    log_message(empty);
});

window.api.ateball.log_message((e, msg) => {
    log_message(msg);
});

// ------------------

var log = document.querySelector('#webview-debug-console');
['log','debug','info','warn','error'].forEach(function (verb) {
    console[verb] = (function (method, verb, log) {
        return function () {
            method.apply(console, arguments);
            var msg = document.createElement('span');
            msg.classList.add("log-object", verb);
            msg.innerHTML += Array.prototype.slice.call(arguments).map((item, idx) => { return (typeof(item) === 'object') ? JSON.stringify(item) : item } ).join(' ');
            log.prepend(msg);
        };
    })(console[verb], verb, log);
});

const log_level = {0 : "info", 1 : "log", 2 : "warn", 3 : "error"};
webview.addEventListener('console-message', (e) => {
    console[log_level[e.level]](e.message);
});

const debug = document.querySelector("#ateball-debug-console");
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
