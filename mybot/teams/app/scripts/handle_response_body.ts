export function handleResponse(responseBody: any, requestHeaders: any) {
	if (responseBody && Array.isArray(responseBody.eventMessages) && responseBody.eventMessages.length > 0) {
		for (const event of responseBody.eventMessages) {
			console.debug("handle: event:", event, requestHeaders)
			if (event.type === 'EventMessage' && event.resource && event.resourceType === 'NewMessage') {
				let { resource } = event
				if (resource.lastMessage) {
					resource = resource.lastMessage
				}
				let messageText
				const sentTime = resource.composetime
				const receivedTime = resource.originalarrivaltime
				const from = resource.imdisplayname
				const toId = resource.to
				// Other types: messagetype: "Control/Typing", contenttype: "Application/Message"
				if (resource.messagetype === 'Text' && resource.contenttype === 'text') {
					messageText = resource.content
				} else if (resource.messagetype === 'RichText/Html' && resource.contenttype === 'text') {
					// TODO Remove HTML.
					messageText = resource.lastMessage.content
				}
				const response = getResponse(messageText, from)
				if (response) {
					sendMessage(from, response, toId, requestHeaders)
				}
			}
		}
	}
}

function getResponse(messageText: string, from: string): string | undefined {
	// TODO Look into handling rich text with markdown/HTML. Might need to send a different message type.
	if (/^(hello|hey|hi)\b.{0,10}/i.test(messageText)) {
		const firstName = (from || "").split(' ')[0]
		return `🤖 This is an automated response: Hey ${firstName}, what's up?`
	}
	return undefined
}

function sendMessage(imdisplayname: string, messageText: string, toId: string, requestHeaders: any) {
	console.debug(`onhello/sendMessage: Replying \"${messageText}\" to \"${imdisplayname}\".`)
	// This was mostly copied from watching the Network tab in the browser.
	const url = `https://teams.microsoft.com/api/chatsvc/amer/v1/users/ME/conversations/${toId}/messages`
	const body = {
		content: messageText,
		messagetype: 'Text',
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
		// TODO Maybe it could be teams.live.com?
		referrer: 'https://teams.microsoft.com/_',
		referrerPolicy: 'strict-origin-when-cross-origin',
		body: JSON.stringify(body),
		method: 'POST',
		mode: 'cors',
		credentials: 'include'
	})
}