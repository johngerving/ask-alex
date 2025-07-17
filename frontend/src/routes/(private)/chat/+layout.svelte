<script lang="ts">
	import Chat from '$lib/components/chat/chat.svelte';
	import * as Sidebar from '$lib/components/ui/sidebar';
	import type { LayoutProps } from './$types';
	import AppSidebar from '$lib/components/chat/app-sidebar.svelte';
	import ProfileDropdown from '$lib/components/ui/profile-dropdown/profile-dropdown.svelte';

	let { data, children }: LayoutProps = $props();
	let user = $derived(data.user);
	let chats = $derived(data.chats);
	let chatId = $derived(data.chatId);

	$inspect(user);

	let chatHistory = $derived(data.chatHistory);
</script>

<Sidebar.Provider class="h-full w-full">
	<AppSidebar chatsPromise={chats} />
	<Sidebar.Trigger class="m-2 h-10 w-10" />
	<div class="flex h-full w-full flex-col">
		<header class="flex w-full items-center justify-end p-4">
			<ProfileDropdown {user} />
		</header>
		<main class="w-full flex-grow overflow-hidden pb-2">
			{#if user}
				<Chat {chatId} bind:chatHistory />
			{:else}
				<p>Not authenticated</p>
			{/if}
		</main>
	</div>
	{@render children()}
</Sidebar.Provider>
