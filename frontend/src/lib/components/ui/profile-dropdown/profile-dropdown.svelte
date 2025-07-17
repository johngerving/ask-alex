<script lang="ts">
	import * as DropdownMenu from '$lib/components/ui/dropdown-menu/index.js';
	import * as Avatar from '$lib/components/ui/avatar/index.js';
	import type { User } from '$lib/types/user';
	import { PUBLIC_BACKEND_URL } from '$env/static/public';

	let { user }: { user: User } = $props();

	let initial = $derived.by(() => {
		return user.email.length > 0 ? user.email.charAt(0).toUpperCase() : '';
	});
</script>

<DropdownMenu.Root>
	<DropdownMenu.Trigger>
		<Avatar.Root>
			<Avatar.Fallback>{initial}</Avatar.Fallback>
		</Avatar.Root>
	</DropdownMenu.Trigger>
	<DropdownMenu.Content>
		<DropdownMenu.Group>
			<DropdownMenu.Label>{user.email}</DropdownMenu.Label>
			<DropdownMenu.Separator />
			<DropdownMenu.Item class="p-0">
				<a href="/" class="w-full px-4 py-2 text-left">Home</a>
			</DropdownMenu.Item>
			<DropdownMenu.Item class="p-0">
				<form class="w-full" action={`${PUBLIC_BACKEND_URL}/auth/logout`} method="post">
					<button class="w-full px-4 py-2 text-left">Log Out</button>
				</form>
			</DropdownMenu.Item>
		</DropdownMenu.Group>
	</DropdownMenu.Content>
</DropdownMenu.Root>
