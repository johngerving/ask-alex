<script lang="ts">
	import Button from '$lib/components/ui/button/button.svelte';
	import * as Card from '$lib/components/ui/card/index.js';
	import { ChatBubble } from '$lib/components/ui/chatbubble';
	import { ChatTextarea } from '$lib/components/ui/chattextarea';

	let messages: Message[] = $state([
		{
			content: "Test"
		}
	])
</script>

<main class="flex h-full w-full items-center justify-center p-3">
	<Card.Root class="flex h-full w-full flex-col">
		<Card.Header>
			<Card.Title>Chat</Card.Title>
		</Card.Header>
		<Card.Content class="flex-grow pb-2 pt-2">
			<div class="h-full w-full">
				{#each messages as message}
					<ChatBubble message={message}/>
				{/each}
			</div>
		</Card.Content>
		<Card.Footer class="flex items-end gap-2">
			<ChatTextarea class="h-11 min-h-11 resize-none" />
			<Button class="h-11">Send</Button>
		</Card.Footer>
	</Card.Root>
</main>
