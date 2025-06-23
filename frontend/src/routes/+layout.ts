export const ssr = false;

import { getChats } from '$lib/utils/chat/getChats';
import type { LayoutLoad } from './$types';
import { getUser } from '$lib/utils/chat/getUser';
import { redirect } from '@sveltejs/kit';

export const load: LayoutLoad = async ({ fetch }) => {
	let user;

	try {
		user = await getUser(fetch);
	} catch (error) {
		console.error(error);
		redirect(303, '/login');
	}

	return {
		user,
		chats: getChats(fetch)
	};
};
