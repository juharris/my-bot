import { browser } from 'webextension-polyfill-ts'
import { Rule, RuleSettings } from './rules/rules'

export async function handleResponse(url: string, responseBody: any, requestHeaders: any) {
	console.debug("onhello: url:", url)
	// Get the rules each time in case they get updated.
	let { rules } = await browser.storage.local.get('rules')
	console.debug("onhello: rules:", rules)
	if (rules === undefined) {
		const results = await browser.storage.sync.get('rules')
		if (results === undefined || results.rules === undefined) {
			console.debug("onhello: no rules found.")
			return
		}
		rules = results.rules
	}
	let rulesSettings: RuleSettings
	try {
		rulesSettings = JSON.parse(rules)
	} catch (err) {
		console.warn("onhello: Error parsing rules. Open the extension options to correct them.", err)
		return
	}
	for (const settings of rulesSettings.apps) {
		if (settings === undefined || settings.urlPattern === undefined || !(new RegExp(settings.urlPattern, 'i').test(url))) {
			return
		}
		// Handle Teams response.
		if (responseBody && Array.isArray(responseBody.eventMessages) && responseBody.eventMessages.length > 0) {
			for (const event of responseBody.eventMessages) {
				console.debug("handle: event:", event, requestHeaders)
				if (event.type === 'EventMessage' && event.resource && event.resourceType === 'NewMessage') {
					let { resource } = event
					if (resource.lastMessage) {
						resource = resource.lastMessage
					}
					let messageText
					if (resource.composetime) {
						const sentTime = new Date(resource.composetime)
						// Check if it was sent in the last 1 minute.
						if (new Date().getTime() - sentTime.getTime() > 60 * 1000) {
							continue
						}
					}
					// const receivedTime = resource.originalarrivaltime
					const from = resource.imdisplayname
					const toId = resource.to
					// Other types: messagetype: "Control/Typing", contenttype: "Application/Message"
					if (resource.messagetype === 'Text' && resource.contenttype === 'text') {
						messageText = resource.content
					} else if (resource.messagetype === 'RichText/Html' && resource.contenttype === 'text') {
						messageText = resource.content
						if (messageText) {
							// Get rid of HTML tags.
							// There are fancier ways to do this but they can cause issues if they try to render themselves.
							messageText = messageText.replace(/<[^>]+>/g, '')
						}
					}
					if (messageText) {
						console.debug(`onhello/handleResponse: Got \"${messageText}\" from \"${from}\".`)
						const response = getResponse(from, messageText, settings.rules)
						if (response) {
							sendMessage(from, response, toId, requestHeaders)
						}
					}
				}
			}
		}
	}
}

export class Response {
	constructor(
		public text: string,
		public messageType: 'Text' | 'RichText/Html' = 'Text') { }
}

export function replaceResponseText(text: string, from: string): string {
	const firstName = (from || "").split(' ')[0]
	const result = text.replace(/{{\s*FROM_FIRST_NAME\s*}}/g, firstName)
	text = text.replace(/{{\s*FROM\s*}}/g, from)
	return result
}

export function getResponse(from: string, messageText: string, rules: Rule[]): Response | undefined {
	for (const rule of rules) {
		if (rule.messageExactMatch === messageText
			|| (rule.messagePattern !== undefined
				&& new RegExp(rule.messagePattern, rule.regexFlags).test(messageText))) {
			const responseText = replaceResponseText(rule.response, from)
			return new Response(responseText, 'RichText/Html')
		}
	}
	return undefined
}

function sendMessage(imdisplayname: string, response: Response, toId: string, requestHeaders: any) {
	console.debug(`onhello/sendMessage: Replying \"${response.text}\" to \"${imdisplayname}\".`)
	// This was mostly copied from watching the Network tab in the browser.
	const url = `https://teams.microsoft.com/api/chatsvc/amer/v1/users/ME/conversations/${toId}/messages`
	const body = {
		content: response.text,
		messagetype: response.messageType,
		contenttype: 'text',
		amsreferences: [],
		clientmessageid: `${new Date().getTime()}${Math.floor(Math.random() * 1E4)}`,
		imdisplayname,
		properties: {
			importance: "",
			subject: null
		}
	}
	// TODO Look into retrieving some fields from other messages.
	return fetch(url, {
		headers: {
			accept: 'json',
			'accept-language': "en-US,en;q=0.9",
			authentication: requestHeaders.Authentication,
			behavioroverride: requestHeaders.BehaviorOverride,
			clientinfo: requestHeaders.ClientInfo,
			'content-type': 'application/json',
			"sec-ch-ua": "\" Not A;Brand\";v=\"99\", \"Chromium\";v=\"90\", \"Microsoft Edge\";v=\"90\"",
			'sec-ch-ua-mobile': '?0',
			'sec-fetch-dest': 'empty',
			'sec-fetch-mode': 'cors',
			'sec-fetch-site': 'same-origin',
			'x-ms-client-env': requestHeaders['x-ms-client-env'],
			'x-ms-client-type': requestHeaders['x-ms-client-type'],
			'x-ms-client-version': requestHeaders['x-ms-client-version'],
			'x-ms-scenario-id': requestHeaders['x-ms-scenario-id'],
			'x-ms-session-id': requestHeaders['x-ms-session-id'],
			'x-ms-user-type': requestHeaders['x-ms-user-type'],
		},
		// Maybe it could also be teams.live.com?
		// referrer: 'https://teams.microsoft.com/_',
		referrerPolicy: 'strict-origin-when-cross-origin',
		body: JSON.stringify(body),
		method: 'POST',
		mode: 'cors',
		credentials: 'include'
	})
}