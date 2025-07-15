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
		'bg-background placeholder:text-muted-foreground ease flex h-11 w-full rounded-md text-base outline-none transition duration-100 disabled:cursor-not-allowed disabled:opacity-50',
		className
	)}
	bind:value
	{...restProps}
></textarea>
