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
        if (target && target.classList.contains("d-none")) {
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

var targeting_menu = document.querySelector("#targeting-menu");
var targetables = document.querySelectorAll(".targetable");

var suit_selection = document.querySelector("#realtime-suit");
suit_selection.addEventListener("click", (e) => {
    var target = (e.target.classList.contains("suit")) ? e.target : e.target.closest(".suit");
    if (target) {
        window.api.get_state().then((s) => {
            if (s.ateball.game.suit == null) {
                if (!targeting_menu.classList.contains("selecting")) {
                    targeting_menu.classList.add("selecting");
                } else {
                    targetables.forEach((elem) => {
                        if (!elem.disabled) {
                            elem.classList.toggle("target", elem.classList.contains(target.value));
                        }
                    });
                }
            }
        });
    }
});

var ball_targeting_menu = document.querySelector("#realtime-ball-status");
ball_targeting_menu.addEventListener("click", (e) => {
    if (e.target.classList.contains("targetable")) {
        if (!targeting_menu.classList.contains("selecting")) {
            targeting_menu.classList.add("selecting");
        } else {
            let toggle = !e.target.classList.contains("target");
            toggleTargetable(e.target, toggle);
        }
    }
});

ball_targeting_menu.addEventListener("mousedown", (e) => {
    if (targeting_menu.classList.contains("selecting") && e.target.classList.contains("targetable")) {
        let toggle = !e.target.classList.contains("target");

        ball_targeting_menu.onmousemove = (e) => {
            let targetable = document.elementFromPoint(e.clientX, e.clientY);
            toggleTargetable(targetable, toggle);
        }
    }
});

ball_targeting_menu.addEventListener("mouseup", (e) => {
    ball_targeting_menu.onmousemove = null;
});

const toggleTargetable = (targetable, toggle) => {
    var available_targets = document.querySelectorAll(".targetable.target").length;
    if (available_targets > 1) {
        targetable.classList.toggle("target", toggle);
    } else if (toggle) {
        targetable.classList.toggle("target", toggle);
    }
}

var confirm_targeting_btn = document.querySelector("#confirm-targeting-btn");
confirm_targeting_btn.addEventListener("click", (e) => {
    var ball_targets = Array.from(targetables).filter((t) => t.classList.contains("ball")).reduce((a, v) => ({ ...a, [v.id]: v.classList.contains("target") && !v.disabled}), {}) ;

    targeting_menu.classList.remove("selecting");

    window.api.ateball.game.update_targets({
        "targets" : ball_targets
    });
});

var cancel_targeting_btn = document.querySelector("#cancel-targeting-btn");
cancel_targeting_btn.addEventListener("click", (e) => {
    window.api.get_state().then((s) => {
        targetables.forEach(targetable => {
            let target = false;

            if (s.ateball.game.suit == null && s.ateball.game.targets == null) {
                targetable.classList.add("target");
            } else {
                if (s.ateball.game.suit == null) {
                    target = s.ateball.game.targets[targetable.id] && !targetable.disabled;
                } else if (s.ateball.game.targets == null) {
                    target = targetable.classList.contains(s.ateball.game.suit) && !targetable.disabled;
                } else {
                    target = targetable.classList.contains(s.ateball.game.suit) && s.ateball.game.targets[targetable.id] && !targetable.disabled;
                }

                targetable.classList.toggle("target", target);
            }
        });
    });
    

    targeting_menu.classList.remove("selecting");
});

var ateball_webview = document.querySelector("#ateball-webview");
ateball_webview.addEventListener("mousedown", (e) => {
    window.api.get_state().then((s) => {
        if (s !== null && s.ateball.game.started) {
            ateball_webview.addEventListener("mousemove", mousemove);
        }
    });
});

ateball_webview.addEventListener("mouseup", (e) => {
    ateball_webview.classList.remove("crosshair");
    ateball_webview.removeEventListener("mousemove", mousemove);
});
ateball_webview.addEventListener("mouseout", (e) => {
    ateball_webview.classList.remove("crosshair");
    ateball_webview.removeEventListener("mousemove", mousemove);
});

const mousemove = (e) => {
    ateball_webview.classList.add("crosshair");
    webview.send("mousemove", {x: e.clientX, y: e.clientY});
}

var image_options_menu = document.querySelector("#image-options-menu");
image_options_menu.addEventListener("change", (e) => {
    var data = get_realtime_config();
    window.api.ateball.game.realtime.configure(data);
});

var pending_cancel = document.querySelector("#pending-cancel");
pending_cancel.addEventListener("click", (e) => {
    window.api.ateball.game.cancel();
});

var game_stop_press_n_hold; const game_stop_press_n_hold_duration = 1000;
var game_stop = document.querySelector("#game-stop-btn");
game_stop.addEventListener("mousedown", (e) => {
    if (!game_stop.classList.contains("pending") && !game_stop.classList.contains("running")) {
        game_stop.classList.add("pending");
        game_stop.style.animationDuration = `${(game_stop_press_n_hold_duration  / 1000)}s`;
        game_stop_press_n_hold = setTimeout(() => {
            game_stop.classList.remove("pending");
            toggleButtonSpinner(game_stop, true);
            window.api.ateball.game.stop();
        }, game_stop_press_n_hold_duration)
    }
});

game_stop.addEventListener("mouseup", (e) => {
    game_stop.classList.remove('pending');
    clearTimeout(game_stop_press_n_hold);
});

var ateball_stop_press_n_hold; const ateball_stop_press_n_hold_duration = 1000;
var stop = document.querySelector("#ateball-stop");
stop.addEventListener("mousedown", (e) => {
    window.api.get_state().then((s) => {
        if (!(s.ateball.game.pending || s.ateball.game.started)) {
            toggleButtonSpinner(stop, true);
            window.api.ateball.stop();
        } else {
            if (!stop.classList.contains("pending") && !stop.classList.contains("running")) {
                stop.classList.add("pending");
                stop.style.animationDuration = `${(ateball_stop_press_n_hold_duration  / 1000)}s`;
                ateball_stop_press_n_hold = setTimeout(() => {
                    stop.classList.remove("pending");
                    toggleButtonSpinner(stop, true);
                    window.api.ateball.stop();
                }, ateball_stop_press_n_hold_duration)
            }
        }
    });
});

stop.addEventListener("mouseup", (e) => {
    stop.classList.remove('pending');
    clearTimeout(ateball_stop_press_n_hold);
});

const get_realtime_config = () => {
    // formdata doesnt return disabled fields
    return data = {
        "balls" : {
            "solid" : document.querySelector("input[name='draw_solid']").checked,
            "stripe" : document.querySelector("input[name='draw_stripe']").checked
        },
        "table" : {
            "walls" : document.querySelector("input[name='draw_walls']").checked,
            "holes" : document.querySelector("input[name='draw_holes']").checked,
            "background" : document.querySelector("input[name='draw_background']").checked
        }
    };
}

const toggleButtonSpinner = (elem, state) => {
    state ? elem.classList.add("running") : elem.classList.remove("running");
    if (!state) {
        stop.classList.remove("pending");
    }
}

const loading_overlay = document.querySelector("#loading-overlay");

const toggleGUIElements = () => {
    window.api.get_state().then((s) => {
        // enable/disable everything else
        if (s !== null) {
            toggleWebviewControls(s);
            toggleAteballControls(s);
            toggleGamemodeControls(s);
            toggleGameControls(s);

            webview.style.display = (s.webview.formatted) ? "flex" : "none";
            loading_overlay.style.display = (s.webview.loaded) ? "none" : "block";
        }
    });
}

const toggleWebviewControls = (s) => {
    webview.style.pointerEvents = (!s.ateball.game.started) ? "" : "none";
    webview.style.cursor = (!s.ateball.game.started) ? "" : "not-allowed";
}

const toggleAteballControls = (s) => {
    start.disabled = s.process.started;
    start.style.width = s.process.started ? "0" : "";

    stop.disabled = !s.process.started;
    stop.style.width = !s.process.started ? "0" : "";
}

const gamemode_controls = document.querySelector("#gamemode-controls");

const gamemode_pending_overlay = document.querySelector("#gamemode-pending-overlay");
const gamemodes = Array.from(gamemode_controls.querySelectorAll(".gamemode"));

const toggleGamemodeControls = (s) => {
    var interact = s.webview.loaded && s.process.started && s.process.connected && s.webview.menu && s.webview.menu == "/en/game";

    gamemode_controls.disabled = !interact;
    gamemode_pending_overlay.style.display = (s.ateball.pending) ? "block" : "none";

    gamemodes.forEach(function(elem) {
        if (interact && !s.ateball.game.started) {
            if (s.webview.logged_in) {
                elem.disabled = (elem.id == "guest_gamemode");
                elem.style.display = (elem.id == "guest_gamemode") ? "none" : "block";
            } else {
                elem.disabled = (elem.id != "guest_gamemode");
                elem.style.display = "block";
            }
        } else {
            elem.disabled = true;
        }
    });
}

const game_controls = document.querySelector("#game-controls");

const turn_timer = document.querySelector("#turn-timer");
const turn_num = document.querySelector("#turn-timer #turn-num");

const suits = Array.from(document.querySelectorAll(".suit"));
const ball_indicators = Array.from(document.querySelectorAll(".pool-ball-indicator"));

const toggleGameControls = (s) => {
    var interact = s.webview.loaded && s.process.started && s.process.connected && s.webview.menu && s.webview.menu == "/en/game" && s.ateball.game.started;

    game_controls.disabled = !interact;

    if (s.ateball.game.round.started) {
        turn_num.textContent = s.ateball.game.round.turn_num;
        turn_timer.classList.add("start");
        turn_timer.classList.remove("pending");
    } else {
        turn_timer.classList.add("pending");
        turn_timer.classList.remove("start");
    }

    suit_selection.style.display = (s.ateball.game.suit !== undefined) ? "" : "none";
    if (!interact) {
        targeting_menu.classList.remove("selecting");
    }

    if (s.ateball.game.suit !== undefined) {
        suits.forEach((elem) => {
            if (interact && s.ateball.game.suit != null) {
                if (!elem.classList.contains(s.ateball.game.suit)) {
                    elem.classList.remove("selected");
                    elem.style.width = 0;
                } else {
                    elem.classList.add("selected");
                    elem.style.width = "";
                }
            } else {
                elem.classList.remove("selected");
                elem.style.width = "";
            }
        });
    }

    ball_indicators.forEach((elem) => {
        if (interact) {
            if (!(elem.id in s.ateball.game.balls) || s.ateball.game.suit === undefined) {
                elem.disabled = true;
                elem.classList.remove("target");
                elem.classList.remove("targetable");
            } else {
                if (s.ateball.game.suit !== undefined) {
                    // suit is null or solid/stripe
                    
                    if (!targeting_menu.classList.contains("selecting")) {
                        let is_target = true; let is_targetable = true;

                        if (s.ateball.game.suit != null) {
                            is_targetable = elem.classList.contains(s.ateball.game.suit) && !s.ateball.game.balls[elem.id].pocketed;
                            is_target = is_targetable && ((s.ateball.game.targets != null) ? s.ateball.game.targets[elem.id] : s.ateball.game.balls[elem.id].target);
                        } else {
                            is_targetable = !s.ateball.game.balls[elem.id].pocketed;
                            is_target = is_targetable && ((s.ateball.game.targets != null) ? s.ateball.game.targets[elem.id] : s.ateball.game.balls[elem.id].target);
                        }

                        elem.disabled = s.ateball.game.balls[elem.id].pocketed;
                        elem.classList.toggle("target", is_target);
                        elem.classList.toggle("targetable", is_targetable);
                    }
                } else {
                    elem.disabled = true;
                    elem.classList.remove("target");
                    elem.classList.remove("targetable");
                }
            }
        } else {
            elem.disabled = false;
            elem.classList.add("target");
            elem.classList.add("targetable");
        }
    });
}

setInterval(toggleGUIElements, 1000 / 10);
