import type { UIMessage } from 'ai';

export default function ChatMessage({ message }: { message: UIMessage }) {
  const isUser = message.role === 'user';

  let parts: any[] = [];

  // Case 1: New AI SDK format
  if (Array.isArray(message.parts)) {
    parts = message.parts;
  }

  // Case 2: content is already array (image/text structured)
  else if (Array.isArray(message.content)) {
    parts = message.content;
  }

  // Case 3: content is plain string
  else if (typeof message.content === 'string') {
    parts = [{ type: 'text', text: message.content }];
  }

  const textParts = parts.filter(p => p.type === 'text');
  const imageParts = parts.filter(p => p.type === 'image');

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div
        className={`max-w-[80%] px-4 py-2.5 rounded-2xl text-sm leading-relaxed whitespace-pre-wrap space-y-2 ${
          isUser
            ? 'bg-foreground text-background rounded-br-md'
            : 'bg-zinc-100 dark:bg-zinc-800 rounded-bl-md'
        }`}
      >
        {textParts.map((part, index) => (
          <p key={`text-${index}`}>{part.text}</p>
        ))}

        {imageParts.map((part, index) => (
          <img
            key={`img-${index}`}
            src={part.image}
            alt="Uploaded"
            className="rounded-xl max-w-xs"
          />
        ))}
      </div>
    </div>
  );
}