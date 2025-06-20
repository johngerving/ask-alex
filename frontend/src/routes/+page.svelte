<script lang="ts">
	import { Chat } from '$lib/components/chat';
	import type { PageProps } from './$types';

	let { data }: PageProps = $props();
	let user = $derived(data.user);
</script>

<main class="h-full w-full pb-2">
	{#if user}
		<Chat />
	{:else}
		<p>Not authenticated</p>
	{/if}
</main>
