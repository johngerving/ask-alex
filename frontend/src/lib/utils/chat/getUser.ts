import { PUBLIC_BACKEND_URL } from '$env/static/public';
import type { FetchFunction } from '$lib/types/fetchFunction';

export const getUser = async (fetch: FetchFunction) => {
	const response = await fetch(`${PUBLIC_BACKEND_URL}/user`, {
		method: 'GET',
		credentials: 'include'
	});

	if (!response.ok) {
		throw new Error('Failed to fetch user');
	}

	const user = await response.json();

	return user;
};
