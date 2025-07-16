<script lang="ts">
	import * as Sidebar from '$lib/components/ui/sidebar/index.js';
	import * as DropdownMenu from '$lib/components/ui/dropdown-menu/index.js';
	import * as Dialog from '$lib/components/ui/dialog/index.js';
	import { useSidebar } from '$lib/components/ui/sidebar/context.svelte.js';
	import FamiconsTrashOutline from '~icons/famicons/trash-outline';
	import UisEllipsisH from '~icons/uis/ellipsis-h';
	import type { Chat } from '$lib/types/chat';
	import { deleteChat } from '$lib/utils/chat/deleteChat';
	import { cn } from '$lib/utils';
	import { page } from '$app/state';
	import { Button } from '$lib/components/ui/button';
	import { invalidateAll, preloadData } from '$app/navigation';

	let { chatsPromise }: { chatsPromise: Promise<Chat[]> } = $props();

	let chats: Chat[] | undefined = $state(undefined);

	$effect(() => {
		chatsPromise.then((chatsResult) => {
			chats = chatsResult;
		});
	});

	let selectedChatId = $derived(page.params.chatId);

	const sidebar = useSidebar();
</script>

{#if chats === undefined}
	<p>Loading chats...</p>
{:else}
	{#each chats as chat (chat.id)}
		<Sidebar.MenuItem
			style="list-style: none;"
			class={cn('rounded-md', chat.id === selectedChatId ? 'bg-accent' : '')}
		>
			<a
				onclick={() => {
					if (page.url.pathname === '/chat') invalidateAll();
				}}
				href={`/chat/${chat.id}`}
				class="w-full"
			>
				<Sidebar.MenuButton class="px-4 py-3"><span>{chat.title}</span></Sidebar.MenuButton>
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
					class="rounded-lg p-0"
					side={sidebar.isMobile ? 'bottom' : 'right'}
					align={sidebar.isMobile ? 'end' : 'start'}
				>
					<Dialog.Root>
						<Dialog.Trigger class="w-full">
							<DropdownMenu.Item onSelect={(e) => e.preventDefault()} class="w-full">
								<span class="text-muted-foreground w-full p-1">
									<FamiconsTrashOutline class="text-muted-foreground inline h-full align-middle" />
									Delete Chat</span
								>
							</DropdownMenu.Item>
						</Dialog.Trigger>
						<Dialog.Content>
							<Dialog.Header>
								<Dialog.Title>Delete chat?</Dialog.Title>
								<Dialog.Description>This chat will be permanently deleted.</Dialog.Description>
							</Dialog.Header>
							<Dialog.Footer>
								<Dialog.Close>
									<Button variant="secondary">Cancel</Button>
								</Dialog.Close>
								<Button
									variant="destructive"
									onclick={async () => {
										await deleteChat(chat.id);
									}}
								>
									Delete
								</Button>
							</Dialog.Footer>
						</Dialog.Content>
					</Dialog.Root>

					<!-- <button onclick={async () => await deleteChat(chat.id)}>

						</button> -->
				</DropdownMenu.Content>
			</DropdownMenu.Root>
		</Sidebar.MenuItem>
	{/each}
{/if}
<!-- <p class="text-destructive">Error loading chats: {error.message}</p> -->
