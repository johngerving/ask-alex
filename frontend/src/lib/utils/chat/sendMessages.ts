import { type Message } from '$lib/types/message';

import { PUBLIC_BACKEND_URL } from '$env/static/public';
import type { Tool } from '$lib/types/toolCall';

export const sendMessages = async (
	messages: (Message | Tool)[],
	chatId: string,
	fns: {
		onStart: () => void;
		onUpdateContent: (delta: string) => void;
		onUpdateReasoning: (delta: string) => void;
		onToolCall: (toolCall: Tool) => void;
		onFinish: (response: string) => void;
		onError: (error: string) => void;
	}
) => {
	try {
		const messagesCopy = messages.slice();

		fns.onStart();

		// Make a request to the RAG endpoint
		const res = await fetch(`${PUBLIC_BACKEND_URL}/chat/messages`, {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json'
			},
			credentials: 'include',
			body: JSON.stringify({ message: messagesCopy[messagesCopy.length - 1], chatId }) // Pass in query as body of request
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
					try {
						const delta = parseData(data);
						response += delta;
						fns.onUpdateContent(delta);
					} catch (error) {
						console.error('Error parsing delta:', error);
					}
				} else if (eventType === 'reasoning') {
					try {
						const reasoningDelta = parseData(data);
						fns.onUpdateReasoning(reasoningDelta);
					} catch (error) {
						console.error('Error parsing reasoning:', error);
					}
				} else if (eventType === 'tool_call') {
					try {
						const dataObj = JSON.parse(data);
						if (
							!Object.hasOwn(dataObj, 'id') ||
							!Object.hasOwn(dataObj, 'name') ||
							!Object.hasOwn(dataObj, 'kwargs')
						) {
							throw new Error('Tool call data is missing required properties');
						}
						const toolCall: Tool = {
							id: dataObj.id,
							name: dataObj.name,
							kwargs: dataObj.kwargs
						};
						fns.onToolCall(toolCall);
					} catch (error) {
						console.error('Error parsing tool call:', error);
					}
				} else if (eventType === 'response') {
					try {
						const finalResponse = parseData(data);
						fns.onFinish(finalResponse);
						return;
					} catch (error) {
						console.error('Error parsing response:', error);
					}
				} else if (eventType === 'error') {
					try {
						const errorMessage = parseData(data);
						fns.onError(errorMessage);
					} catch (error) {
						console.error('Error parsing error message:', error);
					}
				}

				eventType = '';
				data = '';
			}
		}

		// for (let i = 0; i < 100; i++) {
		// 	fns.onUpdateContent('hello ');
		// }
	} catch (error: any) {
		fns.onError(error);
	}
};

export const parseData = (data: string): string => {
	let obj = {};
	try {
		obj = JSON.parse(data);
	} catch (error: any) {
		throw new Error(`Invalid JSON string: ${error.message}`);
	}

	if (typeof obj == 'object' && 'v' in obj) {
		if (typeof obj.v !== 'string') {
			throw new Error('"v" property is not a string');
		}
		return obj.v;
	} else {
		throw new Error('Data missing "v" property or is not a valid object');
	}
};
