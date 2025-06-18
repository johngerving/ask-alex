<script lang="ts">
	import { Chat } from '$lib/components/chat';
	import type { PageProps } from './$types';

	let { data }: PageProps = $props();
	let user = $derived(data.user);
</script>

<main class="w-full flex-grow overflow-hidden pb-2">
	{#if user}
		<Chat />
	{:else}
		<p>Not authenticated</p>
	{/if}
</main>
