/**
 * Scrolls to the bottom of a node. 
 * @param node The element to scroll to the bottom of.
 * @param messages Messages to update (this argument only exists to let Svelte listen to the state of messages).
 * @returns  
 */
export function scrollToBottom(node: Element, messages: Message[]) {
    const scroll = () =>
        node.scroll({
            top: node.scrollHeight,
            behavior: 'smooth'
        });
    scroll();

    // Return an update to the scroll to animate it
    return { update: scroll };
}