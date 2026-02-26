'use client';

import { useState } from 'react';
import MessageList from './components/MessageList';
import ChatInput from './components/ChatInput';

export default function Chat() {
  const [messages, setMessages] = useState<any[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  // 🟢 Handle text message
  const handleTextSubmit = async (e: any) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMessage = { role: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    const formData = new FormData();
    formData.append('text', input);

    const res = await fetch('/api/chat', {
      method: 'POST',
      body: formData,
    });

    const data = await res.json();

    setMessages(prev => [
      ...prev,
      { id: crypto.randomUUID() ,role: 'assistant', content: data.text },
    ]);

    setIsLoading(false);
  };

  // 🟢 Handle image upload
  const handleImageUpload = async (file: File) => {
    setIsLoading(true);

    // Show image in UI
    const imageUrl = URL.createObjectURL(file);

    setMessages(prev => [
      ...prev,
      {
        id: crypto.randomUUID(),
        role: 'user',
        content: imageUrl,
        isImage: true,
      },
    ]);

    const formData = new FormData();
    formData.append('file', file);

    const res = await fetch('/api/chat', {
      method: 'POST',
      body: formData,
    });

    const data = await res.json();

    setMessages(prev => [
      ...prev,
      { id: crypto.randomUUID(),role: 'assistant', content: data.text },
    ]);

    setIsLoading(false);
  };

  return (
    <div className="flex flex-col h-dvh max-w-2xl mx-auto">
      <header className="p-4 text-center text-sm font-medium border-b border-zinc-200 dark:border-zinc-800">
        AI vs Real Image Detector
      </header>

      <MessageList messages={messages} />

      <ChatInput
        input={input}
        setInput={setInput}
        isLoading={isLoading}
        onSubmit={handleTextSubmit}
        onImageUpload={handleImageUpload}
      />
    </div>
  );
}
