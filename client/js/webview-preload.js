const { ipcRenderer } = require('electron');

var iframe = null;
var canvas = null;

ipcRenderer.on('mousemove', (e, data) =>{
    if (!canvas) { return; }

    try {
        var event = new MouseEvent("mousemove", {
            view: window,
            bubbles: true,
            cancelable: true,
            clientX: data.x,
            clientY: data.y
        });
        canvas.dispatchEvent(event);
    } catch (e) {
        console.log("couldn't mouse move");
    }
});

ipcRenderer.on('mousedown', (e, data) =>{
    if (!canvas) { return; }

    try {
        var event = new MouseEvent("mousedown", {
            view: window,
            bubbles: true,
            cancelable: true,
            clientX: data.x,
            clientY: data.y
        });
        canvas.dispatchEvent(event);
    } catch (e) {
        console.log("couldn't mouse down");
    }
});

ipcRenderer.on('mouseup', (e, data) =>{
    if (!canvas) { return; }

    try {
        var event = new MouseEvent("mouseup", {
            view: window,
            bubbles: true,
            cancelable: true,
            clientX: data.x,
            clientY: data.y
        });
        canvas.dispatchEvent(event);
    } catch (e) {
        console.log("couldn't mouse up");
    }
});

window.addEventListener("DOMContentLoaded", () => {
    format();
});

const format = () => {
    try {
        if (location.host == "me.miniclip.com") {
            if (location.pathname == "/login/pool") {
                document.querySelectorAll("header, footer, .page-header, #form-fb-login").forEach((elem) => {
                    elem.remove();
                });
                
                waitForElement(".login-form").then(() => {
                    document.querySelector(".login-form a[href='/forgot-password']").parentElement.remove();
            
                    document.querySelector("#login-btn").parentElement.style.width = "100%";
                    document.querySelector("#login-btn").style.width = "100%";

                    ipcRenderer.send("webview-loaded", location.pathname, localStorage.getItem("accessToken"));
                });
            }
        } else if (location.pathname == "/en/game" || (location.pathname == "/login" && location.search)) {
            document.querySelectorAll("header, footer, .sidebar-overlay, .sidebar").forEach((elem) => {
                elem.style.display = "none";
            });
    
            waitForElement(".play-area").then(() => {
                var game_page = document.querySelector(".game-page");
                game_page.style.padding = "unset";

                var play_area = document.querySelector(".play-area");

                var play_area_p1 = play_area.parentElement;
                play_area_p1.style.display = "flex";
                play_area_p1.style.width = "fit-content";
                play_area_p1.style.position = "relative";
                play_area_p1.style.margin = "0";

                Array.from(play_area_p1.children).forEach((elem) => {
                    if (elem === play_area) { return; }
                    elem.remove();
                })
                
                iframe = play_area.querySelector("#iframe-game");
                iframe.onload = async () => {
                    var iframe_doc = iframe.contentDocument || iframe.contentWindow.document;
                    var style = document.createElement('style');
    
                    ipcRenderer.invoke('webview-format', "iframe").then((data) => {
                        iframe_doc.head.appendChild(style);
                        style.appendChild(document.createTextNode(data));
                    }).then(() => {
                        if (location.pathname == "/en/game") {
                            waitForElement("#loadingBox", iframe_doc, false).then(() => {
                                ipcRenderer.send("webview-loaded", location.pathname, localStorage.getItem("accessToken"));

                                canvas = iframe_doc.querySelector("#engine");
                                var events = ["mouseup", "mousedown", "mousemove"];
                                events.forEach((ev) => {
                                    canvas.addEventListener(ev, (e) => {
                                        ipcRenderer.sendToHost(ev, { x: e.x, y: e.y});
                                    });
                                });
                            });
                        }
                    });
                }

                document.querySelector("header").remove();
                document.querySelector("footer").remove();
                document.querySelector(".sidebar-overlay").remove();
                document.querySelector(".sidebar").remove();
            });
        }
	} catch (error) {
		console.error(error);
	} finally {
        ipcRenderer.send("webview-formatted");
    }
}

const waitForElement = (selector, _document=document, to_be_visible=true) => {
    return new Promise(resolve => {
        if (_document.querySelector(selector)) {
            if (to_be_visible && _document.querySelector(selector).offsetParent != null) {
                return resolve(_document.querySelector(selector));
            }
        }

        const observer = new MutationObserver(mutations => {
            var elem = _document.querySelector(selector);
            if (elem) {
                if (to_be_visible && elem.offsetParent != null) {
                    resolve(elem);
                    observer.disconnect();
                } else if (!to_be_visible && elem.offsetParent == null) {
                    resolve(elem);
                    observer.disconnect();
                }
            }
        });

        observer.observe(_document.body, {
            childList: true,
            subtree: true
        });
    });
}