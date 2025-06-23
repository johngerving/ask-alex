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

	for (const id of data) {
		if (typeof id !== 'number') throw new Error('Invalid chat ID type');

		chats.push({
			id
		});
	}

	return chats;
}
