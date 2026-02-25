'use client';

import { FormEvent } from 'react';

export default function ChatInput({
  input,
  setInput,
  onSubmit,
  isLoading,
  onImageUpload,
}: {
  input: string;
  setInput: (v: string) => void;
  onSubmit: (e: FormEvent) => void;
  isLoading: boolean;
  onImageUpload: (file: File) => void;
}) {
  return (
    <form
      onSubmit={onSubmit}
      className="flex gap-2 p-4 border-t border-zinc-200 dark:border-zinc-800"
    >
      <input
        type="file"
        accept="image/*"
        onChange={e => {
          if (e.target.files && e.target.files[0]) {
            onImageUpload(e.target.files[0]);
          }
        }}
        className="text-sm"
      />

      <input
        className="flex-1 bg-transparent px-4 py-2.5 rounded-xl border border-zinc-200 dark:border-zinc-800 outline-none text-sm"
        value={input}
        onChange={e => setInput(e.currentTarget.value)}
        placeholder="Ask about the result..."
      />

      <button
        type="submit"
        disabled={!input.trim() || isLoading}
        className="px-4 py-2.5 rounded-xl bg-foreground text-background text-sm font-medium disabled:opacity-30"
      >
        {isLoading ? '...' : 'Send'}
      </button>
    </form>
  );
}