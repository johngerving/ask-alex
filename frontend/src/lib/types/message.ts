export enum MessageType {
	User = "user",
	Assistant = "assistant"
}

export interface Message {
	content: string;
	type: MessageType;
}
