<script lang="ts">
	import { PUBLIC_BACKEND_URL } from '$env/static/public';
	import Chat from '$lib/components/chat/chat.svelte';
	import * as Sidebar from '$lib/components/ui/sidebar';
	import type { LayoutProps } from './$types';
	import AppSidebar from '$lib/components/chat/app-sidebar.svelte';
	import { page } from '$app/state';
	import { invalidateAll, onNavigate } from '$app/navigation';

	let { data, children }: LayoutProps = $props();
	let user = $derived(data.user);
	let chats = $derived(data.chats);
	let chatId = $derived(data.chatId);

	let chatHistory = $derived(data.chatHistory);

	onNavigate(() => {
		console.log('Navigated');
	});
</script>

<Sidebar.Provider class="h-full w-full">
	<AppSidebar chatsPromise={chats} />
	<Sidebar.Trigger class="m-2 h-10 w-10" />
	<main class="h-full w-full pb-2">
		{#if user}
			<Chat {chatId} bind:chatHistory />
		{:else}
			<p>Not authenticated</p>
		{/if}
	</main>

	{@render children()}
</Sidebar.Provider>
