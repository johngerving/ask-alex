<script lang="ts">
	import type { WithElementRef, WithoutChildren } from 'bits-ui';
	import type { HTMLTextareaAttributes } from 'svelte/elements';
	import { cn } from '$lib/utils.js';

	let {
		ref = $bindable(null),
		value = $bindable(),
		class: className,
		...restProps
	}: WithoutChildren<WithElementRef<HTMLTextareaAttributes>> = $props();

	$effect(() => {
		value;

		if (ref instanceof HTMLTextAreaElement) {
			ref.style.height = '44px';

			const style = getComputedStyle(ref);
			const borderTop = parseFloat(style.borderTopWidth);
			const borderBottom = parseFloat(style.borderBottomWidth);

			const newHeight = ref.scrollHeight + borderTop + borderBottom;

			ref.style.height = Math.min(newHeight, 136) + 'px';
		}
	});
</script>

<textarea
	bind:this={ref}
	class={cn(
		'border-input bg-background ring-offset-background placeholder:text-muted-foreground focus-visible:ring-ring ease flex h-11 w-full rounded-md border px-5 py-2 text-base transition duration-100 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50',
		className
	)}
	bind:value
	{...restProps}
></textarea>
