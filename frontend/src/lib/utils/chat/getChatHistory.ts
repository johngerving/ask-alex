import { PUBLIC_BACKEND_URL } from '$env/static/public';
import type { FetchFunction } from '$lib/types/fetchFunction';
import { MessageStatus, type Message } from '$lib/types/message';
import type { Tool } from '$lib/types/toolCall';
import { error } from '@sveltejs/kit';
import { v4 } from 'uuid';

export async function getChatHistory(chatId: number, fetch: FetchFunction): Promise<Message[]> {
	const response = await fetch(`${PUBLIC_BACKEND_URL}/chat/${chatId}/messages`, {
		method: 'GET',
		credentials: 'include'
	});

	if (!response.ok) {
		error(404, 'Chat not found');
	}

	const data = await response.json();
	if (!Array.isArray(data)) {
		error(500, 'Internal server error');
	}

	let messages: Message[] = [];

	for (let i = data.length - 1; i >= 0; i--) {
		let msg = data[i];

		if (!Object.hasOwn(msg, 'type')) {
			console.error('Message type is missing: ', msg);
			error(500, 'Internal server error');
		}

		if (msg.type === 'message') {
			messages.unshift({
				id: v4(),
				status: MessageStatus.Finished,
				toolCalls: [],
				reasoning: '',
				content: msg.content,
				role: msg.role
			});
		}

		if (msg.type === 'tool_call') {
			if (
				!Object.hasOwn(msg, 'name') ||
				!Object.hasOwn(msg, 'kwargs') ||
				!Object.hasOwn(msg, 'id')
			) {
				console.error('Tool call data is missing required properties: ', msg);
				error(500, 'Internal server error');
			}

			if (
				typeof msg.name !== 'string' ||
				typeof msg.name !== 'string' ||
				typeof msg.kwargs !== 'object'
			) {
				console.error('Tool call data has incorrect types: ', msg);
				error(500, 'Internal server error');
			}

			if (messages.length === 0) {
				console.error('Tool call without preceding message: ', msg);
				error(500, 'Internal server error');
			}

			messages[0].toolCalls.unshift({
				id: msg.id,
				name: msg.name,
				kwargs: msg.kwargs
			});
		}
	}

	return messages;
}
