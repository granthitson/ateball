const { ipcRenderer } = require('electron');
  
ipcRenderer.on('format', () => {
    format();
});

window.addEventListener("DOMContentLoaded", async (e) => {
    format();
});

window.addEventListener("message", function (e) {
    try {
        var data = JSON.parse(e.data);
        console.log(data);
    } catch (e) {}
});

function format() {
    try {
        console.log(location.pathname);
        if (location.host == "me.miniclip.com") {
            if (location.pathname == "/login/pool") {
                document.querySelectorAll("header, footer, .page-header, #form-fb-login").forEach((elem) => {
                    elem.remove();
                });
                
                waitForElement(".login-form").then(() => {
                    document.querySelector(".login-form a[href='/forgot-password']").parentElement.remove();
            
                    document.querySelector("#login-btn").parentElement.style.width = "100%";
                    document.querySelector("#login-btn").style.width = "100%";
                });
            }
        } else if (location.pathname == "/en/game" || (location.pathname == "/login" && location.search)) {
            document.querySelectorAll("header, footer, .sidebar-overlay, .sidebar").forEach((elem) => {
                elem.style.display = "none";
            });
    
            waitForElement(".play-area").then(() => {
                var play_area_p1 = document.querySelector(".play-area").parentElement;
                play_area_p1.style.display = "flex";
                play_area_p1.style.width = "fit-content";
                play_area_p1.style.position = "relative";
                play_area_p1.style.margin = "0";
        
                var play_area = document.querySelector(".play-area");
                Array.from(play_area_p1.children).forEach((elem) => {
                    if (elem === play_area) { return; }
                    elem.remove();
                })
                
                var iframe = play_area.querySelector("#iframe-game");
                iframe.onload = async () => {
                    console.log(iframe);
                    var iframe_doc = iframe.contentDocument || iframe.contentWindow.document;
                    var style = document.createElement('style');
    
                    ipcRenderer.invoke('get-css', "iframe").then((data) => {
                        iframe_doc.head.appendChild(style);
                        style.appendChild(document.createTextNode(data));
                    });
                }
            });
        }
	} catch (error) {
		console.log(error);
	} finally {
        // waitForElement(".play-area").then(() => {
        //     document.querySelector("header").remove();
        //     document.querySelector("footer").remove();
        //     document.querySelector(".sidebar-overlay").remove();
        //     document.querySelector(".sidebar").remove();
        // });
    }
}

function waitForElement(selector) {
    return new Promise(resolve => {
        if (document.querySelector(selector)) {
            return resolve(document.querySelector(selector));
        }

        const observer = new MutationObserver(mutations => {
            if (document.querySelector(selector)) {
                resolve(document.querySelector(selector));
                observer.disconnect();
            }
        });

        observer.observe(document.body, {
            childList: true,
            subtree: true
        });
    });
}