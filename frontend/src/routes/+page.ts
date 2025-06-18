export const ssr = false;

import { getUser } from '$lib/utils/chat/getUser';
import { redirect } from '@sveltejs/kit';
import type { PageLoad } from './$types';

export const load: PageLoad = async () => {
	let user;

	try {
		user = await getUser();
	} catch (error) {
		console.error(error);
		redirect(303, '/login');
	}

	return {
		user
	};
};
