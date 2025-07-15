<script lang="ts">
	import * as Sidebar from '$lib/components/ui/sidebar/index.js';
	import { Button } from '$lib/components/ui/button';
	import { createChat } from '$lib/utils/chat/createChat';
	import type { Chat } from '$lib/types/chat';
	import ChatList from './chat-list.svelte';
	import { goto, invalidateAll } from '$app/navigation';

	let { chatsPromise }: { chatsPromise: Promise<Chat[]> } = $props();
</script>

<Sidebar.Root>
	<Sidebar.Header>
		<a href="/chat" class="w-full">
			<Button class="w-full transition-colors duration-200">New Chat</Button>
		</a>
	</Sidebar.Header>
	<Sidebar.Content>
		<Sidebar.Group>
			<Sidebar.GroupLabel>Chats</Sidebar.GroupLabel>
			<ChatList {chatsPromise} />
		</Sidebar.Group>
	</Sidebar.Content>
	<Sidebar.Footer />
</Sidebar.Root>
