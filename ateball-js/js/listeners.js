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

var selects = document.querySelectorAll("select");
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
        window.api.ateball.play(btn.dataset).then(() => {
            toggleGUIElements(get_state());
            document.querySelector("a[data-bs-target='#game-controls']").click()
        });
    });
});

var start = document.querySelector("#ateball-start");
start.addEventListener("click", (e) => {
    toggleAteballStart(true);
    window.api.ateball.start();
});

var cancel = document.querySelector("#pending-cancel");
cancel.addEventListener("click", (e) => {
    document.querySelector("a[data-bs-target='#menu-controls']").click()
    window.api.ateball.cancel();
});

var stop = document.querySelector("#ateball-stop");
stop.addEventListener("click", (e) => {
    toggleAteballStop(true);
    window.api.ateball.stop();
});

const toggleAteballStart = (state) => {
    var start = document.querySelector("#ateball-start");
    start.disabled = !state;
}

const toggleAteballStop = (state) => {
    var start = document.querySelector("#ateball-stop");
    start.disabled = !state;
}

const toggleAteballControls = (state) => {
    if (state) {
        toggleAteballStart(false);
        toggleAteballStop(true);
    } else {
        toggleAteballStart(true);
        toggleAteballStop(false);
    }
}

const toggleGUIElements = (state, parent_id=undefined) => {
	var elem_list = ["button:not(.static)", "input.ateball-input", "select.menu-select"];

	var parent = document.getElementById("controlpanel");
	if (parent_id !== undefined) {
		parent = document.getElementById(parent_id);
	}
	
	if (parent) {
        state.then((s) => {
            console.log(s);
            if (s !== null /*&& s.loaded*/) {
                document.querySelector("#loading-overlay").style.display = (s.menu && s.menu == "/en/game" && s.loaded) ? "none" : "block";
                document.querySelector("#pending-overlay").style.display = (s.ateball.pending) ? "block" : "none"; 
            }

            elem_list.forEach(function(elemName) {
                parent.querySelectorAll(elemName).forEach(function(elem) {
                    if (s !== null && s.loaded) {
                        if (s.process.started && s.process.connected) {
                            if (s.menu && s.menu == "/en/game") {
                                let interact = false;

                                if (s.ateball.pending) {
                                    // disable gamemode selection buttons / enable game controls if started
                                    interact = (elem.closest(".root-menu").id == "game-controls") ? s.ateball.game.started : false;
                                } else {
                                    if (elem.closest(".root-menu").id == "game-controls") {
                                        interact = false;
                                    } else {
                                        if (s.logged_in) {
                                            interact = (elem.id == "guest-btn") ? false : true;
                                        } else {
                                            interact = (elem.id == "guest-btn") ? true : false;
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
}
