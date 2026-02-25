'use client';

import { useEffect, useRef } from 'react';
import type { UIMessage } from 'ai';
import ChatMessage from './ChatMessage';

export default function MessageList({ messages }: { messages: UIMessage[] }) {
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  if (messages.length === 0) {
    return (
      <div className="flex-1 flex items-center justify-center text-zinc-400 text-sm">
        Start a conversation
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto p-4 space-y-3">
      {messages.map(m => (
        <ChatMessage key={m.id} message={m} />
      ))}
      <div ref={endRef} />
    </div>
  );
}
