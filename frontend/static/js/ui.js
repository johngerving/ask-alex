function handleChatEnter(e) {
    // If the enter button is pressed in the chat form, submit it unless shift is being held
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();

        htmx.trigger("#chat-form", "submit")
    }
}

// Add the event listener on page load
document.getElementById("message").addEventListener("keypress", handleChatEnter);

// We need to add the event listener again after htmx swaps the element
document.body.addEventListener("htmx:afterRequest", e => {
    if (e.detail.target.id == "messages") {
        document.getElementById("message").addEventListener("keypress", handleChatEnter)
    }
})