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
        var bet = menu.querySelector("select[name='bet']");
        var username = menu.querySelector("input[name='username']")

        var play_btn = parent_menu.querySelector("button.play-btn");
        if (play_btn.dataset.gamemode !== undefined) {
            play_btn.dataset.gamemode = btn.dataset.gamemode;
        }
        if (play_btn.dataset.location !== undefined) {
            play_btn.dataset.location = location.value;
        }
        play_btn.dataset.bet = (bet != null) ? bet.value : '';
        if (bet == null) {
            delete play_btn.dataset.bet;
        }
        if (play_btn.dataset.username !== undefined) {
            play_btn.dataset.username = (username != null) ? username.value : '';
            if (username == null) {
                delete play_btn.dataset.bet;
            }
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

var inputs = document.getElementsByClassName("ateball-input");
Array.from(inputs).forEach(input => {
    input.addEventListener("keyup", function(e) {
        var parent_menu = input.closest("div.menu");
        var play_btn = parent_menu.querySelector("button.play-btn");
        
        play_btn.dataset.username = input.value;
        if (input.value === undefined || input.value === "") {
            play_btn.disabled = true;
        } else {
            play_btn.disabled = false;
        }
    });
});

var play_btns = document.getElementsByClassName("play-btn");
Array.from(play_btns).forEach(btn => {
    btn.addEventListener("click", function(e) {
        // sendSocketMessage("play", btn.dataset);
        window.api.send_message(btn.dataset);
        toggleGUIElements(null);
    });
});

const toggleAteballControls = (state) => {
    var start = document.querySelector("#ateball-start");
    var stop = document.querySelector("#ateball-stop");

    if (state) {
        start.disabled = true;
        stop.disabled = false;
    } else {
        start.disabled = false;
        stop.disabled = true;
    }
}

const toggleGUIElements = (state, parent_id=undefined) => {
	var elemList = ["button:not(.static)", "input.ateball-input", "select.menu-select"];

	var parent = document.getElementById("controlpanel");
	if (parent_id !== undefined) {
		parent = document.getElementById(parent_id);
	}
	
	if (parent) {
        state.then((s) => {
            elemList.forEach(function(elemName) {
                parent.querySelectorAll(elemName).forEach(function(elem) {
                    if (s !== null && (s.menu && s.menu == "/en/game")) {
                        if (s.logged_in) {
                            let interact = (elem.id == "guest-btn") ? false : true;
                            elem.disabled = !interact;
                        } else {
                            let interact = (elem.id == "guest-btn") ? true : false;
                            elem.disabled = !interact;
                        }
                    } else {
                        elem.disabled = true;
                    }
                });
            });
		});
	}
}
