class AteballMouse {
    constructor(webview) {
        this.webview = webview

        this.current_x = null;
        this.current_y = null;

        this.active_execution = null;
        this.active_interpolation = null;

        this.accepted_events = ["mousemove", "mouseup", "mousedown"];
    }

    target_path(path) {
        console.log("targeting path", path);

        var events = [
            {"type" : "mousemove", "point" : path.start}
        ];

        this.cancel();
        this.execute_mouse_movement(events);
    }

    execute_path(path) {
        console.log("executing path", path);

        var events = [ 
            {"type" : "mousemove", "point" : path.start},
            {"type" : "mousedown", "point" : path.start},
            {"type" : "mousemove", "point" : path.end},
            {"type" : "mouseup", "point" : path.end}
        ];

        this.cancel();
        this.execute_mouse_movement(events);
    }

    execute_mouse_movement(events) {
        // interpolate to destination
        const interpolate_mouse_movement = (move_to, duration=1000, step=10) => {
            return new Promise((resolve, reject) => {
                var move_mouse = () => {
                    let dx = Math.floor((move_to.x - this.current_x) * 100) / 100;
                    let dy = Math.floor((move_to.y - this.current_y) * 100) / 100;
        
                    dx = (Math.abs(dx) > 1) ? Math.floor(dx * .5) : dx;
                    dy = (Math.abs(dy) > 1) ? Math.floor(dy * .5) : dy;
        
                    let from_x = this.current_x + dx;
                    let from_y = this.current_y + dy;
        
                    this.webview.send("mousemove", {
                        x: from_x,
                        y: from_y
                    }).finally(() => {
                        if (this.active_interpolation == null) {
                            reject();
                        }
    
                        if (this.current_x == move_to.x && this.current_y == move_to.y) {
                            this.active_interpolation = null;
                            resolve();
                        } else {
                            this.active_interpolation = setTimeout(move_mouse, duration / step);
                        }
                    }).catch((err)=> {
                        console.log(err);
                    });
                }
        
                this.active_interpolation = setTimeout(move_mouse, duration / step);
            });
        }

        // configure abort controller
        this.active_execution = new AbortController();
        var signal = this.active_execution.signal;

        // abort any interpolation
        signal.addEventListener("abort", () => {
            clearTimeout(this.active_interpolation);
            this.active_interpolation = null;
        });
        
        window.api.get_state().then(async (s) => {
            if (data.id in s.ateball.game.round.data.ball_paths) {
                var path_menu_item = document.querySelector(`.ball_path[data-id='${data.id}']`);
                path_menu_item.click();
            }
    
            this.webview.focus();
    
            for (const e of events) {
                try {
                    if (signal && signal.aborted) {
                        // abort any future events
                        break;
                    }
    
                    if (e.type === "mousemove") {
                        await interpolate_mouse_movement(e.point);
                    } else {
                        await this.webview.send(e.type, e.point);
                    }
                } catch (err) {
                    console.log(err);
                }
            }
            
            if (signal && !signal.aborted) {
                console.log("path executed");
            } else {
                console.log("path execution cancelled")
            }

            this.active_execution = null;
        });
    }

    cancel() {
        if (this.active_execution != null) {
            this.active_execution.abort();
        }
    }
}