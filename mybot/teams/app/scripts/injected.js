import { handleResponse } from './handle_response_body'

(function (xhr) {
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
		// console.debug("onhello: XHR.send")
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
					handleResponse(this.responseURL, responseBody, this._requestHeaders)
				}
			} catch (err) {
				console.debug("onhello: Error reading response.", err)
			}
		})

		return send.apply(this, arguments)
	}

})(XMLHttpRequest)