const fs = require('fs');
const path = require('path');

class Webview {
	constructor(window) {
		this.window = window;
		
		this.state = {
			formatted: false,
            loaded: false,
            menu: null,
            logged_in: null
		};
	}

    format(type) {
        console.log("formatting ", type);

        if (type == "webview") {
			this.state.formatted = false;
			this.state.loaded = false;
		}

        return this.get_css(type);
    }

	get_state() {
		return this.state;
	}

	get_css(type) {
		return new Promise((resolve, reject) => {
			fs.readFile(path.join("./", `client/css/${type}.css`), 'utf8', (err, data) => {
				if (err) { console.log("error getting css: ", err); reject(); };
				resolve(data);
			});
		})
	}
}

module.exports = { Webview: Webview };