/**
 * code in inject.js
 * added "web_accessible_resources": ["injected.js"] to manifest.json
 */
console.debug("contentscript.js")
var s = document.createElement('script')
s.src = chrome.extension.getURL('scripts/injected.js')
s.onload = function () {
    this.remove()
};
(document.head || document.documentElement).appendChild(s)