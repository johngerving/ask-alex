import { goto, invalidateAll } from '$app/navigation';
import { PUBLIC_BACKEND_URL } from '$env/static/public';

export async function deleteChat(chatId: string): Promise<void> {
	const response = await fetch(`${PUBLIC_BACKEND_URL}/chat/${chatId}`, {
		method: 'DELETE',
		credentials: 'include'
	});

	if (!response.ok) {
		throw new Error('Failed to delete chat');
	}

	// Check if the current page is the chat being deleted
	const currentPath = window.location.pathname;
	console.log('Current path:', currentPath);
	if (currentPath.includes(`/chat/${chatId}`)) {
		console.log('Redirecting to home page after chat deletion');
		// Redirect to the home page if the current chat is deleted
		await goto('/chat');
	}

	await invalidateAll();
}
