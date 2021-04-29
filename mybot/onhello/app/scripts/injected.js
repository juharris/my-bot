import { handleResponse } from './handle_response_body'


// This file is JavaScript because it was tricky to override XMLHttpRequest in TypeScript.

// Ideally we would reload the rules for each request, this helps if the rules changed.
// However, this could be expensive and will often not be necessary.
// Users will just have to refresh the tab of a site if they want to use the latest rules.
// Also, you can't access the extension storage directly in this context.

(function (xhr) {
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
		const rules = window._onhelloRules
		console.debug("onhello: XHR.send")
		if (rules === undefined) {
			console.info("onhello: XHR.send: No rules set.")
		} else {
			this.addEventListener('load', function () {
				// const responseHeaders = this.getAllResponseHeaders()
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
						handleResponse(this.responseURL, responseBody, this._requestHeaders, rules)
					}
				} catch (err) {
					console.debug("onhello: Error reading response.", err)
				}
			})
		}

		return send.apply(this, arguments)
	}
})(XMLHttpRequest)