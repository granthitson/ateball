var dropdown_btns = document.querySelectorAll("._btn.dropdown[data-target]");

var menu_btns = document.querySelectorAll(".menu-btn[data-target]");
menu_btns.forEach(menu_btn => {
    menu_btn.addEventListener("click", function(e) {
        var menu = $(menu_btn.dataset.target);
        menu.slideToggle();
        menu_btn.classList.toggle("open");

        dropdown_btns.forEach((dropdown_btn, idx) => {
            if (menu_btn != dropdown_btn) {
                var dropdown_menu = $(dropdown_btn.dataset.target);

                if (dropdown_btn.classList.contains("open")) {
                    var dropdown_content = document.querySelector(`[data-parent='#${dropdown_btn.id}']`);
                    dropdown_menu.slideUp(400, function() {
                        dropdown_btn.classList.remove("open");
                        if (dropdown_content != null) {
                            dropdown_content.classList.add("d-none");
                        }
                    });
                }
            }
        });
    });
});

var sub_menu_btns = document.querySelectorAll(".sub-menu-btn[data-target]");
sub_menu_btns.forEach(sub_menu_btn => {
    sub_menu_btn.addEventListener("click", function(e) {
        var sub_menu = $(sub_menu_btn.dataset.target);

        var sub_menu_content = document.querySelector(`[data-parent='#${sub_menu_btn.id}']`);
        if (sub_menu_content && sub_menu_content.classList.contains("d-none")) {
            sub_menu_content.classList.remove("d-none");
            if (!sub_menu_btn.classList.contains("open")) {
                sub_menu.slideDown();
                sub_menu_btn.classList.add("open");
            }
        } else {
            sub_menu.slideUp(400, function() {
                sub_menu_btn.classList.remove("open");
                sub_menu_content.classList.add("d-none");
            });
        }

        sub_menu_btns.forEach((_sub_menu_btn, idx) => {
            var _sub_menu_content = document.querySelector(`[data-parent='#${_sub_menu_btn.id}']`);
            if (_sub_menu_btn != sub_menu_btn) {
                _sub_menu_content.classList.add("d-none");
                _sub_menu_btn.classList.remove("open");
            }
        });
    });
});

var type_btns = document.querySelectorAll("._btn:not(.play-btn)[data-gamemode]");
type_btns.forEach(type_btn => {
    type_btn.addEventListener("click", function(e) {
        var parent_menu = document.querySelector(type_btn.dataset.target);
        var sub_menu = document.querySelector(`[data-parent='#${type_btn.id}']`);
        var menu = (!sub_menu) ? parent_menu : sub_menu;

        var location = menu.querySelector("select[name='location']");

        var play_btn = parent_menu.querySelector("button.play-btn");
        if (play_btn.dataset.gamemode !== undefined) {
            play_btn.dataset.gamemode = type_btn.dataset.gamemode;
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
        window.api.ateball.play(btn.dataset);
    });
});

var ateball_toggle = document.querySelector("#ateball-toggle");
var ateball_start_btn = document.querySelector("#ateball-start-btn");
ateball_start_btn.addEventListener("click", (e) => {
    toggleButtonSpinner(ateball_start_btn, true);
    window.api.ateball.start();
});

const realtime_controls = document.querySelector("#realtime");

const realtime_stream_table = document.querySelector("#realtime-stream #table");
const realtime_stream_ball_paths = realtime_stream_table.querySelector("#ball-paths");

const change_round = realtime_controls.querySelectorAll("#turn-timer .change-round");
change_round.forEach(change_btn => {
    change_btn.addEventListener("click", (e) => {
        if (e.target.classList.contains("left")) {
            window.api.ateball.game.realtime.round_decrement();
        } else if (e.target.classList.contains("right")) {
            window.api.ateball.game.realtime.round_increment();
        }

        realtime_stream_ball_paths.querySelectorAll(`.ball_path_wrapper`).forEach((e) => { e.remove(); });
    });
});

const ball_path_container = document.querySelector("#ball-path-container");
ball_path_container.addEventListener("click", (e) => {
    var target = (e.target.classList.contains("ball-path")) ? e.target : e.target.parentElement;
    if (target.classList.contains("ball-path")) {
        window.api.ateball.game.round.select_ball_path(target.dataset.id);
    }
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
    window.api.ateball.realtime.configure({ type: e.target.name, value : e.target.checked});

    if (e.target.dataset.dependency) {
        var dependency = document.querySelector(`fieldset#${e.target.dataset.dependency}`);
        if (dependency) {
            dependency.disabled = !e.target.checked;
            dependency.classList.toggle("show", e.target.checked);
        }
    }
});

var pending_cancel = document.querySelector("#pending-cancel");
pending_cancel.addEventListener("click", (e) => {
    window.api.ateball.game.cancel();
});

var game_stop_press_n_hold; const game_stop_press_n_hold_duration = 1000;
var game_stop = document.querySelector("#game-stop");
var game_stop_btn = document.querySelector("#game-stop-btn");
game_stop_btn.addEventListener("mousedown", (e) => {
    if (!game_stop_btn.classList.contains("pending") && !game_stop_btn.classList.contains("running")) {
        game_stop_btn.classList.add("pending");
        game_stop_btn.style.animationDuration = `${(game_stop_press_n_hold_duration  / 1000)}s`;
        game_stop_press_n_hold = setTimeout(() => {
            game_stop_btn.classList.remove("pending");
            toggleButtonSpinner(game_stop_btn, true);
            window.api.ateball.game.stop();
        }, game_stop_press_n_hold_duration)
    }
});

game_stop_btn.addEventListener("mouseup", (e) => {
    game_stop_btn.classList.remove('pending');
    clearTimeout(game_stop_press_n_hold);
});

var ateball_stop_press_n_hold; const ateball_stop_press_n_hold_duration = 1000;
var ateball_stop_btn = document.querySelector("#ateball-stop-btn");
ateball_stop_btn.addEventListener("mousedown", (e) => {
    window.api.get_state().then((s) => {
        if (!(s.ateball.game.pending || s.ateball.game.started)) {
            toggleButtonSpinner(ateball_stop_btn, true);
            window.api.ateball.stop();
        } else {
            if (!ateball_stop_btn.classList.contains("pending") && !ateball_stop_btn.classList.contains("running")) {
                ateball_stop_btn.classList.add("pending");
                ateball_stop_btn.style.animationDuration = `${(ateball_stop_press_n_hold_duration  / 1000)}s`;
                ateball_stop_press_n_hold = setTimeout(() => {
                    ateball_stop_btn.classList.remove("pending");
                    toggleButtonSpinner(ateball_stop_btn, true);
                    window.api.ateball.stop();
                }, ateball_stop_press_n_hold_duration)
            }
        }
    });
});

ateball_stop_btn.addEventListener("mouseup", (e) => {
    ateball_stop_btn.classList.remove('pending');
    clearTimeout(ateball_stop_press_n_hold);
});

const toggleSwapButton = (btn1, btn2, state) => {
    btn1.disabled = !state;
    btn1.classList.toggle("active", state);

    btn2.disabled = state;
    btn2.classList.toggle("active", !state);
}

const toggleButtonSpinner = (elem, state) => {
    elem.classList.toggle("running", state);
    if (!state) {
        elem.classList.remove("pending");
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
            toggleRealtimeControls(s);
            toggleRealtimeInterface(s);

            webview.style.display = (s.webview.formatted) ? "flex" : "none";
            loading_overlay.style.display = (s.webview.loaded) ? "none" : "block";
        }
    });
}

const closeNavigationMenus = (parent=document) => {
    var nav_menu_btns = parent.querySelectorAll("._btn.dropdown[data-target]");
    nav_menu_btns.forEach((nav_menu_btn) => {
        if (nav_menu_btn.classList.contains("open")) {
            nav_menu_btn.click();
        }
    });
}

const toggleWebviewControls = (s) => {
    webview.style.pointerEvents = (!s.ateball.game.started) ? "" : "none";
    webview.style.cursor = (!s.ateball.game.started) ? "" : "not-allowed";
}

const toggleAteballControls = (s) => {
    toggleSwapButton(ateball_stop_btn, ateball_start_btn, s.process.started);
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

const turn_timer = document.querySelector("#turn-timer");
const turn_timer_progress = document.querySelector(".timer-progress")
const turn_num = document.querySelector("#turn-timer #round-num");

const round_decrement = realtime_controls.querySelector("#turn-timer #decrement-round");
const round_increment = realtime_controls.querySelector("#turn-timer #increment-round");

const suits = Array.from(document.querySelectorAll(".suit"));
const ball_indicators = Array.from(document.querySelectorAll(".pool-ball-indicator"));

const toggleRealtimeControls = (s) => {
    var interact = s.webview.loaded && s.process.started && s.process.connected && s.webview.menu && s.webview.menu == "/en/game" 
        && s.ateball.game.started;

    realtime_controls.disabled = !interact;
    realtime_controls.style.display = !interact ? "none" : "flex";

    game_stop_btn.disabled = !interact;
    game_stop.classList.toggle("active", interact);
    ateball_toggle.classList.toggle("active", !interact);

    toggleRoundIndicator(s, interact);
    toggleSuitIndicators(s);
    toggleBallIndicators(s, interact);
}

const toggleRoundIndicator = (s, interact) => {
    var active_turn = (s.ateball.game.turn.active && s.ateball.game.realtime.current_round == -1);

    turn_timer.classList.toggle("start", active_turn);
    turn_timer.classList.toggle("pending", !active_turn);

    turn_num.textContent = (s.ateball.game.realtime.current_round == -1) ? s.ateball.game.round.num : s.ateball.game.realtime.current_round;
    let time_passed = clamp((Date.now() - (s.ateball.game.turn.start_time * 1000)) / (s.ateball.game.turn.total_duration * 1000), 0, 1) * 100;
    turn_timer_progress.style.width = (interact && s.ateball.game.turn.active) ? `${time_passed.toFixed(2)}%` : "0%";
    turn_timer_progress.style.display = (interact && s.ateball.game.realtime.current_round != -1) ? "none" : "";

    round_decrement.style.display = (interact && ((s.ateball.game.round.num != null && s.ateball.game.realtime.current_round == -1) || s.ateball.game.realtime.current_round > 1)) ? "block" : "none";
    round_increment.style.display = (interact && s.ateball.game.realtime.current_round > 0 && s.ateball.game.realtime.current_round <= s.ateball.game.round.num) ? "block" : "none";
}

var suit_selection = document.querySelector("#realtime-suit");
const toggleSuitIndicators = (s, interact) => {
    suit_selection.style.display = (s.ateball.game.suit !== undefined) ? "" : "none";

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
}

const toggleBallIndicators = (s, interact) => {
    ball_indicators.forEach((elem) => {
        if (interact) {
            if (!(elem.id in s.ateball.game.balls) || s.ateball.game.suit === undefined) {
                elem.disabled = true;
                elem.classList.remove("target");
            } else {
                if (s.ateball.game.suit !== undefined) {
                    // suit is null or solid/stripe
                    
                    let is_target = true;

                    if (s.ateball.game.suit != null) {
                        is_target = ((s.ateball.game.targets != null) ? s.ateball.game.targets[elem.id] : s.ateball.game.balls[elem.id].info.target);
                    } else {
                        is_target = ((s.ateball.game.targets != null) ? s.ateball.game.targets[elem.id] : s.ateball.game.balls[elem.id].info.target);
                    }

                    elem.disabled = s.ateball.game.balls[elem.id].info.pocketed;
                    elem.classList.toggle("target", is_target);
                } else {
                    elem.disabled = true;
                    elem.classList.remove("target");
                }
            }
        } else {
            elem.disabled = false;
            elem.classList.add("target");
        }
    });
}

const toggleRealtimeInterface = (s) => {
    var interact = s.webview.loaded && s.process.started && s.process.connected && s.webview.menu && s.webview.menu == "/en/game" 
        && s.ateball.game.started;

    realtime_stream_table.classList.toggle("raw", s.realtime.config.table.raw);

    drawBallClusters(s, interact);

    var is_realtime = (s.ateball.game.turn.active && s.ateball.game.realtime.current_round == -1);
    ball_path_container.disabled = !(interact && (!is_realtime || (is_realtime && !(s.ateball.game.round.state.executing || s.ateball.game.round.state.executed))));

    listBallPaths(s, interact, is_realtime);
    drawSelectedBallPath(s, interact);
}

const drawBallClusters = (s, interact) => {
    if (interact) {
        let ball_clusters = (s.ateball.game.turn.active && s.ateball.game.realtime.current_round == -1) ? s.ateball.game.round.data.ball_clusters : s.ateball.game.realtime.data.ball_clusters;
        
        if ((s.ateball.game.turn.active || s.ateball.game.realtime.current_round != -1) && Object.keys(ball_clusters).length > 0) {
            let show = !s.realtime.config.table.raw && s.realtime.config.balls.clusters.highlight;

            var existing_ball_clusters = realtime_stream_table.querySelectorAll(".ball-cluster");
            existing_ball_clusters.forEach(e => { e.classList.toggle("show", show); });

            var ball_cluster_elems_map = Array.from(existing_ball_clusters).reduce((a, v) => ({ ...a, [v.dataset.identifier]: v }), {});

            var all_clusters = new Set([ ...Object.keys(ball_clusters), ...Object.keys(ball_cluster_elems_map) ]);
            all_clusters.forEach((identifier) => {
                if (identifier in ball_cluster_elems_map && !(identifier in ball_clusters)) {
                    ball_cluster_elems_map[identifier].remove();
                } else if (!(identifier in ball_cluster_elems_map) && identifier in ball_clusters) {
                    var cluster = ball_clusters[identifier];

                    const [min_x, min_y] = cluster.min;
                    const [max_x, max_y] = cluster.max;
                    let width = max_x - min_x;
                    let height = max_y - min_y;

                    var ball_cluster = document.createElementNS("http://www.w3.org/2000/svg", "svg");
                    ball_cluster.classList.add('ball-cluster');
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
    
                    realtime_stream_table.append(ball_cluster);
                }
            });
        } else {
            realtime_stream_table.querySelectorAll(".ball-cluster").forEach(e => { e.remove(); });
        }
    } else {
        realtime_stream_table.querySelectorAll(".vector-wrapper").forEach(e => { e.remove(); });
        realtime_stream_table.querySelectorAll(".ball-cluster").forEach(e => { e.remove(); });
    }
}

const listBallPaths = (s, interact, is_realtime) => {
    var ball_path_elems = Array.from(ball_path_container.querySelectorAll(".ball-path[data-id]"));
    var ball_path_placeholder = ball_path_container.querySelector("#ball-path-placeholder");

    let ball_paths = (!is_realtime) ? s.ateball.game.realtime.data.ball_paths : s.ateball.game.round.data.ball_paths;

    if (interact) {
        var ball_path_elems_map = ball_path_elems.reduce((a, v) => ({ ...a, [v.dataset.id]: v }), {});
        var prev_ball_path = null;

        var all_paths = new Set([ ...Object.keys(ball_paths), ...Object.keys(ball_path_elems_map) ]);
        all_paths.forEach((id) => {
            if (id in ball_path_elems_map && !(id in ball_paths)) {
                ball_path_elems_map[id].remove();
            } else if (!(id in ball_path_elems_map) && id in ball_paths) {
                var ball_path = ball_paths[id];

                var ball_path_item = document.createElement('div');
                ball_path_item.classList.add('row', 'ball-path', 'interact');
                ball_path_item.dataset.id = id;

                var ball_path_difficulty = document.createElement('span');
                ball_path_difficulty.classList.add('col-sm-3', 'difficulty', 'px-1');
                ball_path_difficulty.innerHTML = ball_path.info.difficulty.toFixed(2);

                var ball_path_icon = document.createElement('span');
                ball_path_icon.classList.add('col', 'target_ball', `b${ball_path.target_ball.ball.info.number}`);

                var ball_path_target_hole = document.createElement('span');
                ball_path_target_hole.classList.add('col-sm-auto', 'target_hole', `${ball_path.target_hole.hole.name}`);
                
                ball_path_item.append(ball_path_difficulty);
                ball_path_item.append(ball_path_icon);
                ball_path_item.append(ball_path_target_hole);

                if (!prev_ball_path) {
                    ball_path_container.append(ball_path_item);
                } else {
                    prev_ball_path.after(ball_path_item)
                }
                
                prev_ball_path = ball_path_item;
            }
        });

        ball_path_placeholder.style.display = (Object.keys(ball_paths).length > 0) ? "none" : "";
    } else {
        ball_path_placeholder.style.display = "";
        ball_path_elems.forEach(e => { e.remove(); });
        realtime_stream_ball_paths.querySelectorAll(".ball_path_wrapper").forEach(e => { e.remove(); });
    }
}

const drawSelectedBallPath = (s, interact) => {
    if (interact) {
        var ball_paths = (s.ateball.game.realtime.current_round == -1) ? s.ateball.game.round.data.ball_paths : s.ateball.game.realtime.data.ball_paths;
        var selected_ball_path = (s.ateball.game.realtime.current_round == -1) ? ball_paths[s.ateball.game.round.data.selected_ball_path_id] : ball_paths[s.ateball.game.realtime.data.selected_ball_path_id];
        var user_selected_ball_path = (s.realtime.config.selected_ball_path_id != null) ? ball_paths[s.realtime.config.selected_ball_path_id] : null;

        var ball_path = (user_selected_ball_path != null) ? user_selected_ball_path : selected_ball_path;

        if (ball_path != null) {
            if (ball_path.id in ball_paths) {
                var selected_ball_path_elem = ball_path_container.querySelector(".ball-path.selected");
                if (selected_ball_path_elem) {
                    if (selected_ball_path_elem.dataset.id != ball_path.id) {
                        selected_ball_path_elem.classList.remove("selected");
                    }
                } else {
                    var ball_path_elem = ball_path_container.querySelector(`.ball-path[data-id='${ball_path.id}']`);
                    ball_path_elem.classList.add("selected");
                }

                var drawn_ball_path = realtime_stream_table.querySelector(`.ball_path_wrapper`);
                if (!drawn_ball_path || (drawn_ball_path && drawn_ball_path.dataset.id != ball_path.id)) {
                    if (drawn_ball_path && drawn_ball_path.dataset.id != ball_path.id) {
                        drawn_ball_path.remove();
                    }

                    drawBallPath(ball_path.id, ball_path);
                }
            } 
        } else {
            ball_path_container.querySelectorAll(".ball-path.selected").forEach((e) => { e.classList.remove("selected"); })
            realtime_stream_table.querySelectorAll(".ball_path_wrapper").forEach((e) => { e.remove(); })
        }
    }
}

const drawBallPath = (id, ball_path) => {
    function setTrajectory(vector_wrapper, vector_line, trajectory) {
        vector_wrapper.style.left = `${trajectory.origin.x}px`;
        vector_wrapper.style.top = `${trajectory.origin.y}px`;
        vector_wrapper.style.rotate = (trajectory) ? `${(trajectory.angle - 90)}deg` : "";

        vector_line.style.display = (trajectory) ? "" : "none";
        vector_line.style.height = (trajectory) ? `${(trajectory.radius + vector_line.offsetWidth)}px` : "";
    }

    var ball_path_wrapper = document.createElement('div');
    ball_path_wrapper.classList.add('ball_path_wrapper');
    ball_path_wrapper.dataset.id = id;

    // create cueball wrapper
    var cueball_to_ball_wrapper = document.createElement('div');
    cueball_to_ball_wrapper.classList.add('vector-wrapper');

    // create cueball vector
    var cueball_to_ball_vector = document.createElement('div');
    cueball_to_ball_vector.classList.add('predicted-vector-line');
    cueball_to_ball_wrapper.append(cueball_to_ball_vector);

    setTrajectory(cueball_to_ball_wrapper, cueball_to_ball_vector, ball_path.cueball.trajectory)
    ball_path_wrapper.append(cueball_to_ball_wrapper);

    // create ball wrapper
    var ball_to_hole_wrapper = document.createElement('div');
    ball_to_hole_wrapper.classList.add('vector-wrapper');

    // mark ball target point
    var ball_target_point = document.createElement('div');
    ball_target_point.classList.add('vector-target-point');
    ball_to_hole_wrapper.append(ball_target_point);

    // create ball vector
    var ball_to_hole_vector = document.createElement('div');
    ball_to_hole_vector.classList.add('predicted-vector-line');
    ball_to_hole_wrapper.append(ball_to_hole_vector);

    setTrajectory(ball_to_hole_wrapper, ball_to_hole_vector, ball_path.target_ball.trajectory)
    ball_path_wrapper.append(ball_to_hole_wrapper);

    realtime_stream_ball_paths.append(ball_path_wrapper);
}

// ---

const trackBallPositions = () => {
    const drawVectorLine = (b_name, v_type, vector, draw) => {
        function setVectorLine(vector_wrapper, vector_line) {
            vector_wrapper.style.left = `${vector.origin.x}px`;
            vector_wrapper.style.top = `${vector.origin.y}px`;
            vector_wrapper.style.rotate = (vector) ? `${(vector.angle - 90)}deg` : "";
    
            vector_line.style.display = (vector) ? "" : "none";
            vector_line.style.height = (vector) ? `${(vector.radius + vector_line.offsetWidth)}px` : "";
        }
    
        var vector_container = realtime_stream_table.querySelector("#ball-vectors");
        var vector_wrapper = realtime_stream_table.querySelector(`.vector-wrapper[data-ball='${b_name}'][data-type='${v_type}']`);
        var vector_line = null;
    
        if (draw && vector != null) {
            if (!vector_wrapper) {
                vector_line = document.createElement('div');
                vector_line.classList.add(['vector-line']);
    
                vector_wrapper = document.createElement('div');
                vector_wrapper.classList.add(['vector-wrapper']);
                vector_wrapper.dataset.ball = b_name;
                vector_wrapper.dataset.type = v_type;
    
                vector_wrapper.append(vector_line);
                vector_container.append(vector_wrapper);
            } else {
                vector_line = vector_wrapper.querySelector(".vector-line");
            }
    
            setVectorLine(vector_wrapper, vector_line);
        } else if (vector_wrapper) {
            vector_wrapper.remove();
        }
    }

    const removeAllVectorLine = (name, v_type) => {
        realtime_stream_table.querySelectorAll(`.vector-wrapper[data-ball='${name}'][data-type='${v_type}']`).forEach((e) => { e.remove(); });
    }

    const removeVectorLines = (name) => {
        realtime_stream_table.querySelectorAll(`.vector-wrapper[data-ball='${name}']`).forEach((e) => { e.remove(); });
    }

    window.api.get_state().then((s) => {
        var interact = s.webview.loaded && s.process.started && s.process.connected && s.webview.menu && s.webview.menu == "/en/game" && s.ateball.game.started;
        
        var balls = realtime_stream_table.querySelectorAll("._ball");

        balls.forEach(ball => {
            ball.style.display = (interact) ? "" : "none";

            var b_name = ball.id.replace("_", "");
            var _ball = s.ateball.game.balls[b_name];

            if (interact) {
                if (_ball && !_ball.info.pocketed) {
                    let draw = (_ball.info.suit == null || (_ball.info.suit != null && s.realtime.config.balls[_ball.info.suit]));
                    ball.style.display = (draw) ? "unset" : "";
                    ball.style.left = `${_ball.center.x}px`;
                    ball.style.top = `${_ball.center.y}px`;

                    if (!s.realtime.config.table.raw) {
                        for (const [v_type, vector] of Object.entries(_ball.vectors)) {
                            if (vector != null && (s.ateball.game.realtime.current_round == -1 || (_ball.info.name == "cueball" && v_type != "deflect" && s.ateball.game.realtime.current_round != -1))) {
                                drawVectorLine(b_name, v_type, vector, draw);
                            } else {
                                removeAllVectorLine(b_name, v_type);
                            }
                        }
                    }
                } else {
                    ball.style.display = "none";
                    removeVectorLines(b_name);
                }
            } else {
                ball.style.display = "none";
                removeVectorLines(b_name);
            }
        });
    });
}

setInterval(toggleGUIElements, 1000 / 10);
setInterval(trackBallPositions, 1000 / 30);