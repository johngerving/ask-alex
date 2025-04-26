import { MessageType, type Message } from '$lib/types/message';

import { PUBLIC_RAG_ENDPOINT } from '$env/static/public';

export const sendMessages = async (
	messages: Message[],
	fns: {
		onStart: () => void;
		onUpdate: (response: string) => void;
		onFinish: () => void;
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
			// console.log('VALUE:', JSON.stringify(decoder.decode(value)));
			const groups = decoder.decode(value).split('\r\n\r\n');
			let event = '';
			let data = '';

			for (const group of groups) {
				const lines = group.split('\r\n');
				console.log('LINES:', lines);
				for (const line of lines) {
					if (line.startsWith('event:')) {
						event = line.substring('event: '.length).trim();
					}
					if (line.startsWith('data:')) {
						data = line.substring('data: '.length);
					}
				}

				// console.log(event, data)

				data = data.replaceAll('\n', '\n');

				if (event === 'delta') {
					let deltaObj = {};
					try {
						deltaObj = JSON.parse(data);
						if ('v' in deltaObj) {
							console.log('Delta: ', deltaObj);
							const delta = deltaObj.v;
							response += delta;
						}
					} catch (error) {
						console.error('Error parsing delta:', error);
					}
				}

				event = '';
				data = '';

				fns.onUpdate(response);
			}
		}

		fns.onFinish();
	} catch (error: any) {
		fns.onError(error);
	}
};
