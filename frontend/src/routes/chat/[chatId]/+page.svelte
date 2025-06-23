<script lang="ts">
	import Chat from '$lib/components/chat/chat.svelte';

	let { data } = $props();

	let user = $derived(data.user);
	let chatHistory = $derived(data.chatHistory);

	$inspect(chatHistory);
</script>

<main class="h-full w-full pb-2">
	{#if user}
		<Chat />
	{:else}
		<p>Not authenticated</p>
	{/if}
</main>
