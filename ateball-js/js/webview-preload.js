const { ipcRenderer } = require('electron');

window.addEventListener("DOMContentLoaded", () => {
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

                    ipcRenderer.sendToHost("loaded");
                });
            }
        } else if (location.pathname == "/en/game" || (location.pathname == "/login" && location.search)) {
            document.querySelectorAll("header, footer, .sidebar-overlay, .sidebar").forEach((elem) => {
                elem.style.display = "none";
            });
    
            waitForElement(".play-area").then(() => {
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
                
                var iframe = play_area.querySelector("#iframe-game");
                iframe.onload = async () => {
                    var iframe_doc = iframe.contentDocument || iframe.contentWindow.document;
                    var style = document.createElement('style');
    
                    ipcRenderer.invoke('get-css', "iframe").then((data) => {
                        iframe_doc.head.appendChild(style);
                        style.appendChild(document.createTextNode(data));
                    }).then(() => {
                        if (location.pathname == "/en/game") {
                            waitForElement("#loadingBox", iframe_doc, false).then(() => {
                                ipcRenderer.sendToHost("loaded");
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
        ipcRenderer.sendToHost("formatting-complete");
    }
});

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