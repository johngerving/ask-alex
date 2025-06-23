<script lang="ts">
	import '../app.css';
	import { PUBLIC_BACKEND_URL } from '$env/static/public';
	import Chat from '$lib/components/chat/chat.svelte';
	import * as Sidebar from '$lib/components/ui/sidebar';
	import type { LayoutProps } from './$types';
	import AppSidebar from '$lib/components/chat/app-sidebar.svelte';

	let { data, children }: LayoutProps = $props();
	let chats = $derived(data.chats);

	$inspect(chats);
</script>

<Sidebar.Provider class="h-full w-full">
	<AppSidebar chatsPromise={chats} />
	<Sidebar.Trigger class="m-2 h-10 w-10" />
	{@render children()}
</Sidebar.Provider>
