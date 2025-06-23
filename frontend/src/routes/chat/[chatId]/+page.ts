import type { Message } from '$lib/types/message';
import { getChatHistory } from '$lib/utils/chat/getChatHistory.js';
import { error, type HttpError } from '@sveltejs/kit';

export async function load({ params, fetch }) {
	const chatIdString = params.chatId;

	const chatId = parseInt(chatIdString);

	let chatHistory: Message[] = [];

	chatHistory = await getChatHistory(chatId, fetch);

	return {
		chatHistory: chatHistory
	};
}
