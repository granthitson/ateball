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
    updateFormUI(image_options_menu);
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
            "stripe" : document.querySelector("input[name='draw_stripe']").checked,
            "clusters" : {
                "highlight" : document.querySelector("input[name='highlight_clusters']").checked
            }
        },
        "table" : {
            "raw" : document.querySelector("input[name='draw_raw']").checked,
            "walls" : document.querySelector("input[name='draw_walls']").checked,
            "holes" : document.querySelector("input[name='draw_holes']").checked,
            "background" : document.querySelector("input[name='draw_background']").checked
        }
    };
}

const toggleSwapButton = (btn1, btn2, state) => {
    btn1.disabled = !state;
    btn1.classList.toggle("active", state);

    btn2.disabled = state;
    btn2.classList.toggle("active", !state);
}

const toggleButtonSpinner = (elem, state) => {
    state ? elem.classList.add("running") : elem.classList.remove("running");
    if (!state) {
        stop.classList.remove("pending");
    }
}

const updateFormUI = (form) => {
    var fieldsets = form.querySelectorAll("fieldset");
    fieldsets.forEach((fieldset) => {
        if (fieldset.dataset.dependsOn) {
            var depends_on = document.querySelector(`input[name='${fieldset.dataset.dependsOn}']`);
            fieldset.disabled = !depends_on.checked;
            fieldset.classList.toggle("show", depends_on.checked);
        }
    });
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
            toggleRealtimeInterface(s);

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
    toggleSwapButton(stop, start, s.process.started);
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
    game_controls.style.display = !interact ? "none" : "flex";

    if (s.ateball.game.round.started) {
        turn_num.textContent = s.ateball.game.turn_num;
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
                            is_targetable = elem.classList.contains(s.ateball.game.suit) && !s.ateball.game.balls[elem.id].info.pocketed;
                            is_target = is_targetable && ((s.ateball.game.targets != null) ? s.ateball.game.targets[elem.id] : s.ateball.game.balls[elem.id].info.target);
                        } else {
                            is_targetable = !s.ateball.game.balls[elem.id].info.pocketed;
                            is_target = is_targetable && ((s.ateball.game.targets != null) ? s.ateball.game.targets[elem.id] : s.ateball.game.balls[elem.id].info.target);
                        }

                        elem.disabled = s.ateball.game.balls[elem.id].info.pocketed;
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

const toggleRealtimeInterface = (s) => {
    var interact = s.webview.loaded && s.process.started && s.process.connected && s.webview.menu && s.webview.menu == "/en/game" && s.ateball.game.started;

    var config = get_realtime_config();

    var table_ui = document.querySelector("#realtime #table");
    table_ui.classList.toggle("raw", config.table.raw);

    if (interact) {
        if (s.ateball.game.round.started && s.ateball.game.round.ball_clusters) {
            let show = !config.table.raw && config.balls.clusters.highlight;

            var existing_ball_clusters = table_ui.querySelectorAll(".ball-cluster");
            if (existing_ball_clusters.length > 0) {
                existing_ball_clusters.forEach(e => { e.classList.toggle("show", show); });
            } else {
                for (const [identifier, cluster] of Object.entries(s.ateball.game.round.ball_clusters)) {
                    const [min_x, min_y] = cluster.min;
                    const [max_x, max_y] = cluster.max;
                    let width = max_x - min_x;
                    let height = max_y - min_y;

                    var ball_cluster = document.createElementNS("http://www.w3.org/2000/svg", "svg");
                    ball_cluster.classList.add(['ball-cluster']);
                    ball_cluster.classList.toggle("show", show);
                    ball_cluster.dataset.identifier = identifier;
    
                    ball_cluster.style.top = `${min_y}px`;
                    ball_cluster.style.left = `${min_x}px`;
                    ball_cluster.style.width = `${width}px`;
                    ball_cluster.style.height = `${height}px`;

                    var ball_cluster_line = document.createElementNS("http://www.w3.org/2000/svg","line");
                    ball_cluster_line.setAttribute("x1", 0);
                    ball_cluster_line.setAttribute("y1", 0);
                    ball_cluster_line.setAttribute("x2", "100%");
                    ball_cluster_line.setAttribute("y2", "100%");

                    ball_cluster.append(ball_cluster_line);
    
                    table_ui.append(ball_cluster);
                }
            }
        } else {
            table_ui.querySelectorAll(".ball-cluster").forEach(e => { e.remove(); });
        }
    } else {
        table_ui.querySelectorAll(".vector-wrapper").forEach(e => { e.remove(); });
        table_ui.querySelectorAll(".ball-cluster").forEach(e => { e.remove(); });
    }
}

const trackBallPositions = () => {
    const drawVectorLine = (name, ball, draw) => {
        function setVectorLine(vector_wrapper, vector_line, ball) {
            vector_wrapper.style.left = `${ball.center.x}px`;
            vector_wrapper.style.top = `${ball.center.y}px`;
            vector_wrapper.style.rotate = (ball.vector) ? `${(ball.vector.angle - 90)}deg` : "";
    
            vector_line.style.display = (ball.vector) ? "" : "none";
            vector_line.style.height = (ball.vector) ? `${(ball.vector.radius + vector_line.offsetWidth)}px` : "";
        }
    
        var table_ui = document.querySelector("#realtime #table");
        var vector_wrapper = table_ui.querySelector(`.vector-wrapper[data-ball='${name}']`);
        var vector_line = null;
    
        if (draw && ball != null && ball.vector) {
            if (!vector_wrapper) {
                vector_line = document.createElement('div');
                vector_line.classList.add(['vector-line']);
    
                vector_wrapper = document.createElement('div');
                vector_wrapper.classList.add(['vector-wrapper']);
                vector_wrapper.dataset.ball = name;
    
                vector_wrapper.append(vector_line);
                table_ui.append(vector_wrapper);
            } else {
                vector_line = vector_wrapper.querySelector(".vector-line");
            }
    
            setVectorLine(vector_wrapper, vector_line, ball);
        } else if (vector_wrapper) {
            vector_wrapper.remove();
        }
    }

    var config = get_realtime_config();

    window.api.get_state().then((s) => {
        var interact = s.webview.loaded && s.process.started && s.process.connected && s.webview.menu && s.webview.menu == "/en/game" && s.ateball.game.started;
        
        var table_ui = document.querySelector("#realtime #table");
        var balls = table_ui.querySelectorAll("._ball");

        balls.forEach(ball => {
            ball.style.display = (interact) ? "" : "none";

            if (interact) {
                var name = ball.id.replace("_", "");
                var _ball = s.ateball.game.balls[name];

                if (_ball && !_ball.info.pocketed) {
                    let draw = (_ball.suit == null || (_ball.info.suit != null && config.balls[_ball.info.suit]));
                    ball.style.display = (draw) ? "unset" : "";
                    ball.style.left = `${_ball.center.x}px`;
                    ball.style.top = `${_ball.center.y}px`;

                    if (!config.table.raw) {
                        drawVectorLine(name, _ball, draw);
                    }
                } else {
                    ball.style.display = "none";
                }
            } else {
                ball.style.display = "none";
            }
        });
    });
}

setInterval(toggleGUIElements, 1000 / 10);
setInterval(trackBallPositions, 1000 / 30);