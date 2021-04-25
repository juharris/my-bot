import { browser, WebRequest } from 'webextension-polyfill-ts'

// function logURL(requestDetails: any) {
// 	// console.log("Loading: ", requestDetails);
// }
console.debug("background")

const urls = [
	"https://teams.microsoft.com/*",
	"https://teams.live.com/*",
	"https://wus.notifications.skype.com/*"
]

// browser.webRequest.onHeadersReceived.addListener(
// 	logURL,
// 	{
// 		urls
// 	}
// )

function listener(details) {
	console.debug("details:", details)
	// FIXME filterResponseData isn't available in Chrome/Edge.
	let filter = browser.webRequest.filterResponseData(details.requestId);
	let decoder = new TextDecoder("utf-8");
	let encoder = new TextEncoder();
	filter.ondata = event => {
		const str = decoder.decode(event.data, { stream: true });
		console.debug("str:", str)
	}

	return {};
}

browser.webRequest.onBeforeRequest.addListener(
	listener,
	{ urls }
);