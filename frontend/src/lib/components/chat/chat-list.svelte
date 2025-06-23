<script lang="ts">
	import * as Sidebar from '$lib/components/ui/sidebar/index.js';
	import * as DropdownMenu from '$lib/components/ui/dropdown-menu/index.js';
	import { useSidebar } from '$lib/components/ui/sidebar/context.svelte.js';
	import FamiconsTrashOutline from '~icons/famicons/trash-outline';
	import UisEllipsisH from '~icons/uis/ellipsis-h';
	import type { Chat } from '$lib/types/chat';
	import { deleteChat } from '$lib/utils/chat/deleteChat';

	let { chatsPromise }: { chatsPromise: Promise<Chat[]> } = $props();

	const sidebar = useSidebar();
</script>

{#await chatsPromise}
	<p>Loading chats...</p>
{:then chats}
	{#each chats as chat}
		<Sidebar.MenuItem style="list-style: none;">
			<a href={`/chat/${chat.id}`}>
				<Sidebar.MenuButton class="py-4"><span>New Chat</span></Sidebar.MenuButton>
			</a>
			<DropdownMenu.Root>
				<DropdownMenu.Trigger>
					{#snippet child({ props })}
						<Sidebar.MenuAction showOnHover {...props}>
							<UisEllipsisH />
							<span class="sr-only">More</span>
						</Sidebar.MenuAction>
					{/snippet}
				</DropdownMenu.Trigger>
				<DropdownMenu.Content
					class="rounded-lg"
					side={sidebar.isMobile ? 'bottom' : 'right'}
					align={sidebar.isMobile ? 'end' : 'start'}
				>
					<DropdownMenu.Item>
						<button onclick={async () => await deleteChat(chat.id)}>
							<span class="text-muted-foreground">
								<FamiconsTrashOutline class="text-muted-foreground inline h-full align-middle" />
								Delete Chat</span
							>
						</button>
					</DropdownMenu.Item>
				</DropdownMenu.Content>
			</DropdownMenu.Root>
		</Sidebar.MenuItem>
	{/each}
{:catch error}
	<p class="text-destructive">Error loading chats: {error.message}</p>
{/await}
