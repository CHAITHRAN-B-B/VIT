'use client';

export type ChatMessageType = {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  isImage?: boolean;
};

export default function ChatMessage({
  message,
}: {
  message: ChatMessageType;
}) {
  const isUser = message.role === 'user';

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div
        className={`max-w-[80%] px-4 py-2.5 rounded-2xl text-sm leading-relaxed whitespace-pre-wrap ${
          isUser
            ? 'bg-foreground text-background rounded-br-md'
            : 'bg-zinc-100 dark:bg-zinc-800 rounded-bl-md'
        }`}
      >
        {message.isImage ? (
          <img
            src={message.content}
            alt="Uploaded"
            className="rounded-xl max-w-xs"
          />
        ) : (
          <p>{message.content}</p>
        )}
      </div>
    </div>
  );
}
