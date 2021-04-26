import { handleResponse } from './handle_response_body'

console.debug("scripts/injected.js");

(function (xhr) {

	var XHR = XMLHttpRequest.prototype;

	var open = XHR.open;
	var send = XHR.send;
	// var setRequestHeader = XHR.setRequestHeader;

	XHR.open = function (method, url) {
		// this._method = method
		// this._url = url
		// this._requestHeaders = {};
		// this._startTime = (new Date()).toISOString();

		return open.apply(this, arguments);
	};

	// XHR.setRequestHeader = function (header, value) {
	// 	this._requestHeaders[header] = value;
	// 	return setRequestHeader.apply(this, arguments);
	// };

	XHR.send = function (postData) {
		console.debug("injected XHR.send")
		this.addEventListener('load', function () {
			if (this._url) {
				if (postData) {
					if (typeof postData === 'string') {
						try {
							// here you get the REQUEST HEADERS, in JSON format, so you can also use JSON.parse
							this._requestHeaders = postData;
						} catch (err) {
							console.log('Request Header JSON decode failed, transfer_encoding field could be base64');
							console.log(err);
						}
					} else if (typeof postData === 'object' || typeof postData === 'array' || typeof postData === 'number' || typeof postData === 'boolean') {
						// do something if you need
					}
				}
			}


			// here you get the RESPONSE HEADERS
			// var responseHeaders = this.getAllResponseHeaders();
			try {
				if (this.responseType != 'blob') {
					let responseBody;
					// if (this.response) {
					// 	responseBody = this.response
					// }
					if (this.responseType === '' || this.responseType === 'text') {
						console.debug("using responseText. responseType:", this.responseType, this.responseText)
						responseBody = JSON.parse(this.responseText)
					} else /*if (this.responseType === 'json')*/ {
						console.debug("using response. responseType:", this.responseType, this.response)
						responseBody = this.response
					}
					handleResponse(responseBody)
					// responseText is string or null

					// here you get RESPONSE TEXT (BODY), in JSON format, so you can use JSON.parse
					// var arr = this.responseText;

					// console.log(JSON.parse(this._requestHeaders));
					// console.log(responseHeaders);
					console.log("injected responseBody:", this._url, responseBody)
				}
			} catch (err) {
				console.debug("Error reading response.", err)
			}
		});

		return send.apply(this, arguments);
	};

})(XMLHttpRequest)