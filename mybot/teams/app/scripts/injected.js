console.log("onhello: in injected.js")

import { handleResponse } from './handle_response_body'


// This file is JavaScript because it was tricky to override XMLHttpRequest in TypeScript.

// Ideally we would reload the rules for each request, this helps if the rules changed.
// However, this could be expensive and will often not be necessary.
// Users will just have to refresh the tab of a site if they want to use the latest rules.
// Also, 
let rules = undefined


// async function getRules() {
// 	let { rules } = await browser.storage.local.get('rules')
// 	console.debug("onhello (get_rules): rules:", rules)
// 	if (rules === undefined) {
// 		const results = await browser.storage.sync.get('rules')
// 		if (results === undefined || results.rules === undefined) {
// 			console.debug("onhello: no rules found.")
// 			return undefined
// 		}
// 		rules = results.rules
// 	}
// }
// import {getRules} from './rules/get_rules'
// import { browser } from 'webextension-polyfill-ts'

// chrome.runtime.onMessage.addListener(
// 	function (request, sender, sendResponse) {
// 		console.log(sender.tab ?
// 			"from a content script:" + sender.tab.url :
// 			"from the extension");
// 		rules = request
// 		sendResponse({ farewell: "goodbye" });
// 	}
// );
// getRules().then(rulesSettings => {
// 	// console.debug("onhello: got rules", rulesSettings)
// 	rules = rulesSettings
// });

// window.addEventListener("message", (event) => {
// 	console.log("onhello: got event", event, event.data)
// }, false);

; (function (xhr) {
	console.debug("onhello: setting up xhr")
	const XHR = XMLHttpRequest.prototype

	const open = XHR.open
	const send = XHR.send
	const setRequestHeader = XHR.setRequestHeader

	XHR.open = function (method, url) {
		// this._url = url
		this._requestHeaders = {}

		return open.apply(this, arguments)
	}

	XHR.setRequestHeader = function (header, value) {
		this._requestHeaders[header] = value
		return setRequestHeader.apply(this, arguments)
	}

	XHR.send = function (postData) {
		console.debug("onhello: XHR.send")
		if (rules === undefined) {
			console.info("onhello: XHR.send: No rules set.")
		} else {
			this.addEventListener('load', function () {
				// const responseHeaders = this.getAllResponseHeaders()
				// console.debug(`onhello: URL check ${this._url === this.responseURL}`, this._url, this.responseURL)
				try {
					if (this.responseType != 'blob') {
						let responseBody
						if (this.responseType === '' || this.responseType === 'text') {
							// console.debug("onhello: using responseText. responseType:", this.responseType, this.responseText)
							responseBody = JSON.parse(this.responseText)
						} else /* if (this.responseType === 'json') */ {
							// console.debug("onhello: using response. responseType:", this.responseType, this.response)
							responseBody = this.response
						}
						// console.debug("onhello: responseBody:", this.responseURL, responseBody)
						handleResponse(this.responseURL, responseBody, this._requestHeaders, window._onhelloRules)
					}
				} catch (err) {
					console.debug("onhello: Error reading response.", err)
				}
			})
		}

		return send.apply(this, arguments)
	}
})(XMLHttpRequest)