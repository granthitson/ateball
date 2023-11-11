const waitForElement = (selector, _document=document.body) => {
    return new Promise(resolve => {
        if (_document.querySelector(selector)) {
            if (document.querySelector(selector).offsetParent != null) {
                return resolve(_document.querySelector(selector));
            }
        }

        const observer = new MutationObserver(mutations => {
            var elem = _document.querySelector(selector);
            if (elem) {
                resolve(elem);
                observer.disconnect();
            }
        });

        observer.observe(_document, {
            childList: true,
            subtree: true
        });
    });
}