import { PUBLIC_BACKEND_URL } from '$env/static/public';
import type { FetchFunction } from '$lib/types/fetchFunction';
import { MessageStatus, type Message } from '$lib/types/message';
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

	const messages: Message[] = data.map((msg: any) => {
		if (typeof msg.content !== 'string' || typeof msg.role !== 'string') {
			error(500, 'Internal server error');
		}

		return {
			id: v4(),
			status: MessageStatus.Finished,
			reasoning: '',
			content: msg.content,
			role: msg.role
		};
	});

	return messages;
}
