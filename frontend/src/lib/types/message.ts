export enum MessageType {
	User,
	Assistant
}

export interface Message {
	content: string;
	type: MessageType;
}
