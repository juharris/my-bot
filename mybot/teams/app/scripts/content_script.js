import { browser } from 'webextension-polyfill-ts'

const s = document.createElement('script')
s.src = browser.extension.getURL('scripts/injected.js')
s.onload = function () {
    this.remove()
};
(document.head || document.documentElement).appendChild(s)