import { useState, useRef, useEffect, useCallback } from 'react';
import { Send, Trash2, Bot, User } from 'lucide-react';
import { sendChat } from '../api/client';
import type { ChatMessage } from '../types';

const WELCOME_MESSAGE: ChatMessage = {
  role: 'assistant',
  content:
    '👋 Welcome! I\'m your AI trading assistant for Chinese A-shares. Ask me about market data, stock analysis, backtesting strategies, or portfolio management.',
  timestamp: new Date(),
};

function formatTime(date?: Date): string {
  if (!date) return '';
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

export default function Chat() {
  const [messages, setMessages] = useState<ChatMessage[]>([WELCOME_MESSAGE]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, scrollToBottom]);

  const handleSend = async () => {
    const text = input.trim();
    if (!text || loading) return;

    const userMsg: ChatMessage = { role: 'user', content: text, timestamp: new Date() };
    setMessages((prev) => [...prev, userMsg]);
    setInput('');
    setLoading(true);

    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }

    try {
      const { reply } = await sendChat(text);
      const assistantMsg: ChatMessage = {
        role: 'assistant',
        content: reply,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, assistantMsg]);
    } catch (err) {
      const errorMsg: ChatMessage = {
        role: 'assistant',
        content: `⚠️ Error: ${err instanceof Error ? err.message : 'Failed to get response. Please try again.'}`,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMsg]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      void handleSend();
    }
  };

  const handleClear = () => {
    setMessages([WELCOME_MESSAGE]);
    setInput('');
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value);
    const el = e.target;
    el.style.height = 'auto';
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
  };

  return (
    <div className="flex flex-col -m-6 h-[calc(100vh)] overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-6 py-4 border-b border-[#334155]">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-lg bg-[#3b82f6] flex items-center justify-center">
            <Bot size={20} className="text-white" />
          </div>
          <div>
            <h1 className="text-lg font-semibold text-[#f8fafc]">AI Trading Assistant</h1>
            <p className="text-xs text-[#94a3b8]">Powered by LLM</p>
          </div>
        </div>
        <button
          onClick={handleClear}
          className="flex items-center gap-2 px-3 py-1.5 text-sm text-[#94a3b8] hover:text-[#f8fafc] hover:bg-[#334155] rounded-lg transition-colors cursor-pointer"
        >
          <Trash2 size={16} />
          Clear Chat
        </button>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-6 py-4 space-y-4">
        {messages.map((msg, i) => (
          <div
            key={i}
            className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`flex gap-3 max-w-[75%] ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}
            >
              <div
                className={`w-8 h-8 rounded-full flex items-center justify-center shrink-0 ${
                  msg.role === 'user' ? 'bg-[#3b82f6]' : 'bg-[#334155]'
                }`}
              >
                {msg.role === 'user' ? (
                  <User size={16} className="text-white" />
                ) : (
                  <Bot size={16} className="text-[#94a3b8]" />
                )}
              </div>
              <div>
                <div
                  className={`px-4 py-2.5 rounded-2xl text-sm leading-relaxed whitespace-pre-wrap ${
                    msg.role === 'user'
                      ? 'bg-[#3b82f6] text-white rounded-br-md'
                      : 'bg-[#1e293b] text-[#f8fafc] rounded-bl-md border border-[#334155]'
                  }`}
                >
                  {msg.content}
                </div>
                <p
                  className={`text-[10px] text-[#64748b] mt-1 ${
                    msg.role === 'user' ? 'text-right' : 'text-left'
                  }`}
                >
                  {formatTime(msg.timestamp)}
                </p>
              </div>
            </div>
          </div>
        ))}

        {loading && (
          <div className="flex justify-start">
            <div className="flex gap-3 max-w-[75%]">
              <div className="w-8 h-8 rounded-full flex items-center justify-center shrink-0 bg-[#334155]">
                <Bot size={16} className="text-[#94a3b8]" />
              </div>
              <div className="px-4 py-3 rounded-2xl rounded-bl-md bg-[#1e293b] border border-[#334155]">
                <div className="flex gap-1.5">
                  <span className="w-2 h-2 rounded-full bg-[#94a3b8] animate-bounce [animation-delay:0ms]" />
                  <span className="w-2 h-2 rounded-full bg-[#94a3b8] animate-bounce [animation-delay:150ms]" />
                  <span className="w-2 h-2 rounded-full bg-[#94a3b8] animate-bounce [animation-delay:300ms]" />
                </div>
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="px-6 py-4 border-t border-[#334155]">
        <div className="flex items-end gap-3 bg-[#1e293b] rounded-xl border border-[#334155] focus-within:border-[#3b82f6] transition-colors px-4 py-2">
          <textarea
            ref={textareaRef}
            value={input}
            onChange={handleInputChange}
            onKeyDown={handleKeyDown}
            placeholder="Ask about stocks, strategies, or market analysis..."
            disabled={loading}
            rows={1}
            className="flex-1 bg-transparent text-sm text-[#f8fafc] placeholder-[#64748b] outline-none resize-none max-h-40 disabled:opacity-50"
          />
          <button
            onClick={() => void handleSend()}
            disabled={loading || !input.trim()}
            className="p-2 rounded-lg bg-[#3b82f6] hover:bg-[#2563eb] text-white disabled:opacity-40 disabled:cursor-not-allowed transition-colors cursor-pointer shrink-0"
          >
            <Send size={18} />
          </button>
        </div>
        <p className="text-[10px] text-[#64748b] mt-2 text-center">
          Press Enter to send · Shift+Enter for new line
        </p>
      </div>
    </div>
  );
}
