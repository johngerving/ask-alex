import { goto } from '$app/navigation';
import { PUBLIC_BACKEND_URL } from '$env/static/public';
import type { FetchFunction } from '$lib/types/fetchFunction';
import type { User } from '$lib/types/user';

export const getUser = async (fetch: FetchFunction): Promise<User> => {
	const response = await fetch(`${PUBLIC_BACKEND_URL}/user`, {
		method: 'GET',
		credentials: 'include'
	});

	if (!response.ok) {
		throw new Error('Failed to fetch user');
	}

	const user = await response.json();

	if (!user || !('id' in user) || !('email' in user)) throw new Error('Invalid user data received');

	if (typeof user.id !== 'number' || typeof user.email !== 'string') {
		throw new Error('User data has incorrect types');
	}

	return user as User;
};
