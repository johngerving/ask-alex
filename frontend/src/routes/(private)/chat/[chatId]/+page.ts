import type { Message } from '$lib/types/message';
import { getChatHistory } from '$lib/utils/chat/getChatHistory.js';
import { messageStore } from '$lib/state/messages.svelte.js';

export async function load({ params, fetch }) {
	const chatIdString = params.chatId;

	const chatId = parseInt(chatIdString);

	const chatHistory = await getChatHistory(chatId, fetch);

	return {
		chatId: chatId,
		chatHistory: chatHistory
	};
}
