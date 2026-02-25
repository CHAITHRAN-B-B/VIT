'use client';

import { useChat } from '@ai-sdk/react';
import { useState } from 'react';
import MessageList from './components/MessageList';
import ChatInput from './components/ChatInput';

export default function Chat() {
  const [input, setInput] = useState('');
  const { messages, sendMessage, status } = useChat();
  const isLoading = status === 'streaming' || status === 'submitted';

  const handleImageUpload = async (file: File) => {
    const imageUrl = URL.createObjectURL(file);

    // Show image in chat (UI only)
    sendMessage({
      role: 'user',
      content: [
        { type: 'text', text: 'Please analyze this image.' },
        { type: 'image', image: imageUrl }
      ]
    });

    // Run ViT
    const formData = new FormData();
    formData.append('file', file);

    const res = await fetch('http://localhost:8000/predict', {
      method: 'POST',
      body: formData,
    });

    const data = await res.json();

    // Now trigger Gemini THROUGH useChat
    sendMessage(
      {
        text: 'Generate forensic explanation for the uploaded image.'
      },
      {
        data: { vitResult: data }
      }
    );
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
        onSubmit={e => {
          e.preventDefault();
          if (!input.trim()) return;
          sendMessage({ text: input });
          setInput('');
        }}
        onImageUpload={handleImageUpload}
      />
    </div>
  );
}