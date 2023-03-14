var dropdown_btns = document.querySelectorAll("._btn.dropdown[data-target]");

var menu_btns = document.querySelectorAll(".menu-btn[data-target]");
menu_btns.forEach(btn => {
    btn.addEventListener("click", function(e) {
        var btn_target = btn.dataset.target;
        $(btn_target).slideToggle();
        if (btn.classList.contains("open")) {
            btn.classList.remove("open");
        } else {
            btn.classList.add("open");
        }

        dropdown_btns.forEach((dropdown_btn, idx) => {
            if (btn != dropdown_btn) {
                var id = dropdown_btn.id;
                var dropdown_target = dropdown_btn.dataset.target;

                if (dropdown_btn.classList.contains("open")) {
                    var target = document.querySelector("[data-parent='#" + id + "']");
                    $(dropdown_target).slideUp(400, function() {
                        dropdown_btn.classList.remove("open");
                        if (target != null) {
                            target.classList.add("d-none");
                        }
                    });
                }
            }
        });
    });
});

var sub_menu_btns = document.querySelectorAll(".sub-menu-btn[data-target]");
sub_menu_btns.forEach(btn => {
    btn.addEventListener("click", function(e) {
        var id = btn.id; 
        var btn_target = btn.dataset.target;

        var target = document.querySelector("[data-parent='#" + id + "']");
        if (target.classList.contains("d-none")) {
            target.classList.remove("d-none");
            if (!btn.classList.contains("open")) {
                $(btn_target).slideDown();
                btn.classList.add("open");
            }
        } else {
            $(btn_target).slideUp(400, function() {
                btn.classList.remove("open");
                target.classList.add("d-none");
            });
        }

        sub_menu_btns.forEach((sub_menu_btn, idx) => {
            var id = sub_menu_btn.id; 
            var _target = document.querySelector("[data-parent='#" + id + "']");
            if (btn != sub_menu_btn) {
                _target.classList.add("d-none");
                sub_menu_btn.classList.remove("open");
            }
        });
    });
});

var type_btns = document.querySelectorAll("._btn:not(.play-btn)[data-gamemode]");
type_btns.forEach(btn => {
    btn.addEventListener("click", function(e) {
        var id = btn.id;
        var target = btn.dataset.target;
        
        var parent_menu = document.querySelector(target);
        var sub_menu = document.querySelector("[data-parent='#" + id + "']");
        var menu = (!sub_menu) ? parent_menu : sub_menu;

        var location = menu.querySelector("select[name='location']");

        var play_btn = parent_menu.querySelector("button.play-btn");
        if (play_btn.dataset.gamemode !== undefined) {
            play_btn.dataset.gamemode = btn.dataset.gamemode;
        }
        if (play_btn.dataset.location !== undefined) {
            play_btn.dataset.location = location.value;
        }
    });
});

var selects = document.querySelectorAll("#menu-controls select");
selects.forEach(select => {
    select.addEventListener("change", function(e) {
        var parent_menu = select.closest("div.menu");
        var play_btn = parent_menu.querySelector("button.play-btn");

        switch (select.name) {
            case 'bet':
                play_btn.dataset.bet = select.value;
                break;
            case 'location':
                play_btn.dataset.location = select.value;
                break;
        }
    });
});

var play_btns = document.getElementsByClassName("play-btn");
Array.from(play_btns).forEach(btn => {
    btn.addEventListener("click", function(e) {
        var realtime_config = get_realtime_config();
        window.api.ateball.play({ "game_config" : btn.dataset, "realtime_config" : realtime_config});
    });
});

var start = document.querySelector("#ateball-start");
start.addEventListener("click", (e) => {
    toggleButtonSpinner(start, true);
    window.api.ateball.start();
});

var realtime_menu = document.querySelector("#realtime-menu");
realtime_menu.addEventListener("change", (e) => {
    var data = get_realtime_config();
    window.api.ateball.game.realtime.configure(data);
});

var pending_cancel = document.querySelector("#pending-cancel");
pending_cancel.addEventListener("click", (e) => {
    window.api.ateball.game.cancel();
});

var game_cancel = document.querySelector("#game-cancel-btn");
game_cancel.addEventListener("click", (e) => {
    toggleButtonSpinner(game_cancel, true);
    window.api.ateball.game.cancel();
});

var stop = document.querySelector("#ateball-stop");
stop.addEventListener("click", (e) => {
    toggleButtonSpinner(stop, true);
    window.api.ateball.stop();
});

const get_realtime_config = () => {
    // formdata doesnt return disabled fields
    return data = {
        "image_type" : document.querySelector("select[name='image_type']").value,
        "show_walls" : document.querySelector("input[name='show_walls']").checked,
        "show_holes" : document.querySelector("input[name='show_holes']").checked
    };
}

const toggleButtonSpinner = (elem, state) => {
    state ? elem.classList.add("running") : elem.classList.remove("running");
}

const toggleAteballStart = (state) => {
    var start = document.querySelector("#ateball-start");
    start.disabled = !state;
    start.style.width = !state ? "0" : "";
}

const toggleAteballStop = (state) => {
    var stop = document.querySelector("#ateball-stop");
    stop.disabled = !state;
    stop.style.width = !state ? "0" : "";
}

const toggleAteballControls = (state) => {
    toggleAteballStart(!state);
    toggleAteballStop(state);
}

const toggleGUIElements = () => {
	var elem_list = ["button", "input", "select.menu-select"];

    window.api.get_state().then((s) => {
        if (s !== null) {
            toggleAteballControls(s.process.started);

            document.querySelector("webview").style.display = (s.webview.formatted) ? "flex" : "none";

            document.querySelector("#loading-overlay").style.display = (s.webview.loaded) ? "none" : "block";
            document.querySelector("#pending-overlay").style.display = (s.ateball.pending) ? "block" : "none"; 
            document.querySelector("#game-controls").style.display = (s.ateball.game.started) ? "flex" : "none"; 
        }

        elem_list.forEach(function(elemName) {
            Array.from(document.querySelectorAll(elemName)).filter(el => !el.closest('#debug-controls')).forEach(function(elem) {
                if (s !== null && s.webview.loaded) {
                    if (s.process.started && s.process.connected) {
                        if (s.webview.menu && s.webview.menu == "/en/game") {
                            let interact = false;

                            if (s.ateball.pending || s.ateball.game.started) {
                                // disable gamemode selection buttons / enable game controls if started
                                interact = (elem.closest(".controls").id == "game-controls") ? s.ateball.game.started : false;
                            } else {
                                if (elem.closest(".controls").id == "game-controls") {
                                    interact = false;
                                } else {
                                    if (s.webview.logged_in) {
                                        interact = (elem.id == "guest-btn") ? false : true;
                                        elem.style.display = (elem.id == "guest-btn") ? "none" : "block";
                                    } else {
                                        interact = (elem.id == "guest-btn") ? true : false;
                                        elem.style.display = "block";
                                    }
                                }
                            }

                            elem.disabled = !interact;
                        } else {
                            elem.disabled = true;
                        }
                    } else {
                        elem.disabled = true;
                    }
                } else {
                    elem.disabled = true;
                }
            });
        });
    });
}

setInterval(toggleGUIElements, 1000 / 10);
