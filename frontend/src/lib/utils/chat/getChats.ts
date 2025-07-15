import { PUBLIC_BACKEND_URL } from '$env/static/public';
import type { Chat } from '$lib/types/chat';
import type { FetchFunction } from '$lib/types/fetchFunction';

export async function getChats(fetch: FetchFunction): Promise<Chat[]> {
	const response = await fetch(`${PUBLIC_BACKEND_URL}/chat`, {
		method: 'GET',
		credentials: 'include'
	});

	if (!response.ok) {
		throw new Error('Failed to fetch chats');
	}
	const data = await response.json();

	let chats: Chat[] = [];

	if (!Array.isArray(data)) throw new Error('Invalid response from server');

	for (const chat of data) {
		if (typeof chat !== 'object' || chat === null) {
			throw new Error('Invalid chat object');
		}
		if (!('id' in chat)) throw new Error('Chat object missing id');
		if (typeof chat.id !== 'string') throw new Error('Invalid chat ID type');

		if (!('title' in chat)) throw new Error('Chat object missing title');
		if (typeof chat.title !== 'string') throw new Error('Invalid chat title type');

		chats.push({
			id: chat.id,
			title: chat.title
		});
	}

	return chats;
}
