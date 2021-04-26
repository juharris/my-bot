export function handleResponse(responseBody: any) {
    if (responseBody && Array.isArray(responseBody.eventMessages) && responseBody.eventMessages.length > 0) {
        for (const event of responseBody.eventMessages) {
            console.debug("handle: event:", event)
            if (event.type === 'EventMessage' && event.resource && event.resourceType === 'NewMessage' && event.resource.lastMessage) {
                const { resource } = event
                let messageText, from, sentTime, receivedTime
                sentTime = resource.lastMessage.composetime
                receivedTime = resource.lastMessage.originalarrivaltime
                from = resource.lastMessage.imdisplayname
                // Other types: messagetype: "Control/Typing", contenttype: "Application/Message"
                if (resource.lastMessage.messagetype === 'Text' && resource.lastMessage.contenttype === 'text') {
                    messageText = resource.lastMessage.content
                } else if (resource.lastMessage.messagetype === 'RichText/Html' && resource.lastMessage.contenttype === 'text') {
                    // TODO Remove HTML.
                    messageText = resource.lastMessage.content
                }
            }
        }
    }
}
