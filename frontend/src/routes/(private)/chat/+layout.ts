export const ssr = false;

import { getChats } from '$lib/utils/chat/getChats';
import type { LayoutLoad } from './$types';
import { getUser } from '$lib/utils/chat/getUser';
import { redirect } from '@sveltejs/kit';
import type { Message } from '$lib/types/message';
import { getChatHistory } from '$lib/utils/chat/getChatHistory.js';

export const load: LayoutLoad = async ({ fetch, params }) => {
	let user;

	try {
		user = await getUser(fetch);
	} catch (error) {
		console.error(error);
		redirect(303, '/login');
	}

	let chatId: string = '';
	let chatHistory: Message[] = [];
	if (Object.hasOwn(params, 'chatId') && typeof params.chatId === 'string') {
		chatId = params.chatId;

		chatHistory = await getChatHistory(chatId, fetch);
	}

	return {
		user,
		chats: getChats(fetch),
		chatId: chatId,
		chatHistory: chatHistory
	};
};
