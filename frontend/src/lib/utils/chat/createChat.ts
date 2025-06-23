import { goto, invalidateAll } from '$app/navigation';
import { PUBLIC_BACKEND_URL } from '$env/static/public';
import { redirect } from '@sveltejs/kit';

export async function createChat() {
	const response = await fetch(`${PUBLIC_BACKEND_URL}/chat`, {
		method: 'POST',
		credentials: 'include'
	});

	if (!response.ok) {
		throw new Error('Failed to create chat');
	}

	const data = await response.json();
	if (!data || !data.id) {
		throw new Error('Invalid response from server');
	}

	const chatId = data.id;

	goto(`/chat/${chatId}`, {
		invalidateAll: true
	});
}
