import { handleResponse } from './handle_response_body'

console.debug("scripts/injected.js");

(function (xhr) {

	var XHR = XMLHttpRequest.prototype;

	var open = XHR.open;
	var send = XHR.send;
	var setRequestHeader = XHR.setRequestHeader;

	XHR.open = function (method, url) {
		this._method = method
		this._url = url
		this._requestHeaders = {}

		return open.apply(this, arguments)
	}

	XHR.setRequestHeader = function (header, value) {
		this._requestHeaders[header] = value;
		return setRequestHeader.apply(this, arguments)
	}

	XHR.send = function (postData) {
		console.debug("injected XHR.send")
		this.addEventListener('load', function () {
			// const responseHeaders = this.getAllResponseHeaders()
			try {
				if (this.responseType != 'blob') {
					let responseBody;
					if (this.responseType === '' || this.responseType === 'text') {
						console.debug("using responseText. responseType:", this.responseType, this.responseText)
						responseBody = JSON.parse(this.responseText)
					} else /*if (this.responseType === 'json')*/ {
						console.debug("using response. responseType:", this.responseType, this.response)
						responseBody = this.response
					}
					handleResponse(responseBody, this._requestHeaders)
					console.debug("injected responseBody:", this._url, responseBody)
				}
			} catch (err) {
				console.debug("Error reading response.", err)
			}
		});

		return send.apply(this, arguments);
	};

})(XMLHttpRequest)