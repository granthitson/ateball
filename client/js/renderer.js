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
            [current_x, current_y] = [data.x, data.y];
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

var [current_x, current_y] = [null, null];
var current_path_execution = null;
window.api.ateball.game.round.on_execute_path((e, data) => {
    console.log("executing path", data);

    const interpolate_mouse_movement = (move_to, duration=500, step=10) => {
        return new Promise((resolve, reject) => {
            var move_mouse = () => {
                let dx = Math.floor((move_to.x - current_x) * 100) / 100;
                let dy = Math.floor((move_to.y - current_y) * 100) / 100;
    
                dx = (Math.abs(dx) > 1) ? Math.floor(dx * .5) : dx;
                dy = (Math.abs(dy) > 1) ? Math.floor(dy * .5) : dy;
    
                let from_x = current_x + dx;
                let from_y = current_y + dy;
    
                webview.send("mousemove", {
                    x: from_x,
                    y: from_y
                }).finally(() => {
                    if (current_path_execution == null) {
                        reject();
                    }

                    if (current_x == move_to.x && current_y == move_to.y) {
                        resolve();
                    } else {
                        current_path_execution = setTimeout(move_mouse, duration / step);
                    }
                });
            }
    
            current_path_execution = setTimeout(move_mouse, duration / step);
        });
    }

    window.api.get_state().then((s) => {
        if (data.id in s.ateball.game.round.ball_paths) {
            var path_menu_item = document.querySelector(`.ball_path_wrapper[data-id='${data.id}']`);
            console.log(path_menu_item);
            path_menu_item.click();
        }

        const execute_path_events = [ 
            {"type" : "mousemove", "point" : data.start},
            {"type" : "mousedown", "point" : data.start},
            {"type" : "mousemove", "point" : data.end},
            {"type" : "mouseup", "point" : data.end}
        ];

        webview.focus();

        const execute_path = async (events) => {
            for (const e of events) {
                console.log(e);
                if (e.type === "mousemove") {
                    await interpolate_mouse_movement(e.point);
                } else {
                    await webview.send(e.type, e.point);
                }
            }

            console.log("path executed");
        }
        execute_path(execute_path_events);
    });
});

window.api.ateball.game.on_end((e) => {
    console.log("game ended");
    toggleButtonSpinner(game_stop, false);
    closeNavigationMenus(game_controls);

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
