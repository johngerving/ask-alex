import { MessageType, type Message } from '$lib/types/message';

import { PUBLIC_RAG_ENDPOINT } from '$env/static/public';

export const sendMessages = async (
	messages: Message[],
	fns: {
		onStart: () => void;
		onUpdate: (response: string) => void;
		onFinish: (response: string) => void;
		onError: (error: any) => void;
	}
) => {
	try {
		const messagesCopy = messages.slice();

		fns.onStart();

		// Make a request to the RAG endpoint
		const res = await fetch(PUBLIC_RAG_ENDPOINT, {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json'
			},
			body: JSON.stringify({ messages: messagesCopy }) // Pass in query as body of request
		});

		const reader = res.body?.getReader();
		const decoder = new TextDecoder('utf-8');
		if (!reader) throw new Error('Reader not found for response');

		let response = '';

		while (true) {
			const { done, value } = await reader.read();
			if (done) break;
			const groups = decoder.decode(value).split('\r\n\r\n');
			let eventType = '';
			let data = '';

			for (const group of groups) {
				const lines = group.split('\r\n');
				for (const line of lines) {
					if (line.startsWith('event:')) {
						eventType = line.substring('event: '.length).trim();
					}
					if (line.startsWith('data:')) {
						data = line.substring('data: '.length);
					}
				}

				data = data.replaceAll('\n', '\n');

				if (eventType === 'delta') {
					let deltaObj = {};
					try {
						const delta = parseData(data);
						response += delta;
						fns.onUpdate(response);
					} catch (error) {
						console.error('Error parsing delta:', error);
					}
				} else if (eventType === 'response') {
					const finalResponse = parseData(data);
					fns.onUpdate(finalResponse);
					fns.onFinish(finalResponse);
					return;
				}

				eventType = '';
				data = '';
			}
		}
	} catch (error: any) {
		fns.onError(error);
	}
};

const parseData = (data: string): string => {
	let obj = {};
	obj = JSON.parse(data);
	if ('v' in obj) {
		const value = obj.v as string;
		return value;
	} else {
		throw new Error('Data missing "v" property');
	}
};
