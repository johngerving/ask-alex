<script lang="ts">
	import { ArrowRight, Bot, FileText, Search } from 'lucide-svelte';
	import MdiGithub from '~icons/mdi/github';
</script>

<div class="bg-background text-foreground flex h-full w-full flex-col overflow-x-hidden font-sans">
	<header
		class="border-border bg-background/95 supports-[backdrop-filter]:bg-background/60 sticky top-0 z-40 w-full border-b backdrop-blur"
	>
		<div class="container flex h-14 max-w-screen-2xl items-center justify-between">
			<a href="/" class="mr-6 flex items-center space-x-2">
				<span class="font-bold sm:inline-block">Ask Alex</span>
			</a>
			<nav class="flex items-center gap-4 text-sm">
				<a
					href="/chat"
					class="bg-primary text-primary-foreground hover:bg-primary/90 focus-visible:ring-ring inline-flex items-center justify-center whitespace-nowrap rounded-md px-4 py-2 text-sm font-medium shadow transition-colors focus-visible:outline-none focus-visible:ring-1 disabled:pointer-events-none disabled:opacity-50"
				>
					Get Started
				</a>
				<a
					href="https://github.com/johngerving/ask-alex"
					target="_blank"
					rel="noreferrer"
					class="text-foreground/60 hover:text-foreground/80 transition-colors"
				>
					<MdiGithub class="h-10 w-10" />
				</a>
			</nav>
		</div>
	</header>
	<main class="w-full flex-grow">
		<section class="pb-8 pt-6 md:pb-12 md:pt-10 lg:py-32">
			<div class="mx-auto flex max-w-[64rem] flex-col items-center gap-4 text-center">
				<h1 class="font-serif text-3xl font-bold sm:text-5xl md:text-6xl lg:text-7xl">
					An Research Assistant for Institutional Knowledge
				</h1>
				<p class="text-muted-foreground max-w-[42rem] leading-normal sm:text-xl sm:leading-8">
					Ask Alex helps you get answers from Cal Poly Humboldt's institutional repositories.
					Explore articles and gather insights all in one place.
				</p>
				<div class="">
					<a
						href="/chat"
						class="bg-primary text-primary-foreground hover:bg-primary/90 focus-visible:ring-ring inline-flex items-center justify-center whitespace-nowrap rounded-md px-5 py-2 text-base font-medium shadow transition-colors focus-visible:outline-none focus-visible:ring-1 disabled:pointer-events-none disabled:opacity-50"
					>
						Get Started
					</a>
				</div>
			</div>
		</section>
		<section id="features" class="">
			<div class="container">
				<div class="mx-auto mb-3 flex max-w-[58rem] flex-col items-center gap-4 text-center">
					<h2 class="font-serif text-3xl font-bold leading-[1.1] sm:text-3xl md:text-5xl">
						Features
					</h2>
					<p class="text-muted-foreground max-w-[85%] leading-normal sm:text-lg sm:leading-7">
						This project is an experiment to see how a modern chat-based AI assistant can be used to
						improve research workflows.
					</p>
				</div>
				<div
					class="mx-auto grid justify-center gap-4 sm:grid-cols-2 md:max-w-[64rem] md:grid-cols-3"
				>
					<div class="bg-background relative rounded-lg border p-2">
						<div class="flex h-[180px] flex-col justify-between rounded-md p-6">
							<Bot class="h-12 w-12" />
							<div class="space-y-2">
								<h3 class="font-bold">Conversational AI</h3>
								<p class="text-muted-foreground text-sm">
									Chat with your documents and get human-like answers.
								</p>
							</div>
						</div>
					</div>
					<div class="bg-background relative rounded-lg border p-2">
						<div class="flex h-[180px] flex-col justify-between rounded-md p-6">
							<FileText class="h-12 w-12" />
							<div class="space-y-2">
								<h3 class="font-bold">Document Analysis</h3>
								<p class="text-muted-foreground text-sm">
									Extract key information and summaries from your documents.
								</p>
							</div>
						</div>
					</div>
					<div class="bg-background relative overflow-hidden rounded-lg border p-2">
						<div class="flex h-[180px] flex-col justify-between rounded-md p-6">
							<Search class="h-12 w-12" />
							<div class="space-y-2">
								<h3 class="font-bold">Smart Search</h3>
								<p class="text-muted-foreground text-sm">
									Quickly find relevant information across all your uploaded documents.
								</p>
							</div>
						</div>
					</div>
				</div>
			</div>
		</section>
	</main>
	<footer class="py-6 md:px-8 md:py-0">
		<div class="container flex flex-col items-center justify-between gap-4 md:h-24 md:flex-row">
			<p class="text-muted-foreground text-balance text-center text-sm leading-loose md:text-left">
				© 2024 Ask Alex. All rights reserved.
			</p>
		</div>
	</footer>
</div>
