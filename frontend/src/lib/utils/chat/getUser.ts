import { PUBLIC_BACKEND_URL } from '$env/static/public';

export const getUser = async () => {
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
