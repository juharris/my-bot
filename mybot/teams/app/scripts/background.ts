// import { browser, WebRequest } from 'webextension-polyfill-ts'

// function logURL(requestDetails: any) {
// 	// console.log("Loading: ", requestDetails);
// }
// console.debug("background.ts")

// const urls = [
// 	"https://teams.microsoft.com/*",
// 	"https://teams.live.com/*",
// 	"https://wus.notifications.skype.com/*"
// ]

// browser.webRequest.onHeadersReceived.addListener(
// 	logURL,
// 	{
// 		urls
// 	}
// )

// function listener(details: any) {
// 	console.debug("details:", details)
// Doesn't work in Chrome/Edge (`filterResponseData` is Firefox only)
// 	let filter = browser.webRequest.filterResponseData(details.requestId);
// 	let decoder = new TextDecoder("utf-8");
// 	let encoder = new TextEncoder();
// 	filter.ondata = event => {
// 		const str = decoder.decode(event.data, { stream: true });
// 		console.debug("str:", str)
// 	}

// 	return {};
// }

// browser.webRequest.onBeforeRequest.addListener(
// 	listener,
// 	{ urls }
// );