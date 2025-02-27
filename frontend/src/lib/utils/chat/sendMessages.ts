import { MessageType, type Message } from "$lib/types/message";

import { PUBLIC_RAG_ENDPOINT } from "$env/static/public";

export const sendMessages = async (messages: Message[]): Promise<Message> => {
    // Make a request to the RAG endpoint
    const res = fetch(PUBLIC_RAG_ENDPOINT, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ messages: messages }) // Pass in query as body of request
    });

    // Wait for response
    const json = await (await res).json();
    
    if(!('response' in json))
        throw new Error("Invalid response received: 'response' field not present");
        
    const response: string = json.response;

    const message: Message = {
        content: response,
        type: MessageType.Assistant
    } 

    return message
}