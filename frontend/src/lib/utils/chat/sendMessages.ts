import { MessageType, type Message } from "$lib/types/message";

import { PUBLIC_RAG_ENDPOINT } from "$env/static/public";

export const sendMessages = async (messages: Message[], onResponse: (r: string) => void) => {
    // Make a request to the RAG endpoint
    const res = await fetch(PUBLIC_RAG_ENDPOINT, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ messages: messages }) // Pass in query as body of request
    });

    const reader = res.body?.getReader();
    const decoder = new TextDecoder('utf-8');
    if(!reader)
        throw new Error("Reader not found for response")

    let response = "";

    while (true) {
        const { value, done } = await reader.read();
        if (done) 
            break;
        const lines = decoder.decode(value).split('\n');
        let event = "";
        let data = "";

        for(const line of lines) {
            if(line.startsWith("event:")) {
                event = line.substring("event: ".length).trim()
            }
            if(line.startsWith("data:")) {
                data = line.substring("data: ".length)
            }
        }

        // console.log(event, data)

        if (event === "delta") {
            let delta = JSON.parse(data);
            if("v" in delta) {
                response += delta.v;
            }
        }

        onResponse(response)
    }
}