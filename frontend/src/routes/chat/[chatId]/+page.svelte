<script lang="ts">
	import Chat from '$lib/components/chat/chat.svelte';

	let { data } = $props();

	let user = $derived(data.user);
	let chatId = $derived(data.chatId);

	let chatHistory = $derived(data.chatHistory);

	$inspect(chatHistory);
</script>

<main class="h-full w-full pb-2">
	{#if user}
		<Chat {chatId} bind:chatHistory />
	{:else}
		<p>Not authenticated</p>
	{/if}
</main>
