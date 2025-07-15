<script lang="ts">
	import { slide } from 'svelte/transition';
	import MynauiTool from '~icons/mynaui/tool';
	import type { Tool } from '$lib/types/toolCall';

	let { toolCall }: { toolCall: Tool } = $props();

	const toolNameMapping = {
		query_knowledge_base: 'Searching knowledge base',
		call_document_search_agent: 'Searching documents',
		analyze_documents: 'Analyzing documents'
	};

	let toolName = $derived(
		toolCall.name in toolNameMapping
			? toolNameMapping[toolCall.name as keyof typeof toolNameMapping]
			: toolCall.name
	);

	let kwargsDisplay = $derived(
		Object.entries(toolCall.kwargs).map(([key, value]) => {
			return `${key}: ${JSON.stringify(value)}`;
		})
	);

	let open = $state(false);
</script>

<div in:slide class="border-border w-full overflow-hidden rounded-[10px] border">
	<button
		onclick={() => (open = !open)}
		class="hover:bg-muted flex w-full items-center gap-2 p-1 transition-all"
	>
		<div class="border-border h-fit w-fit rounded-[6px] border p-1">
			<MynauiTool class="text-muted-foreground h-5 w-5" />
		</div>
		<h2 class="text-muted-foreground text-sm italic">{toolName}</h2>
	</button>
	{#if open}
		<div transition:slide class="border-border border-t p-2 text-sm">
			{#each kwargsDisplay as display}
				<p class="text-muted-foreground text-left">{display}</p>
			{/each}
		</div>
	{/if}
</div>
