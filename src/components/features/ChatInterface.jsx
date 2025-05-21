// src/components/features/ChatInterface.jsx
import { useState, useRef, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { Send, ChevronDown, ChevronUp, ExternalLink } from 'lucide-react';
import Button from '../common/Button';
import { useAuth } from '../../context/AuthContext';
import { useToast } from '../common/Toast';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const ChatInterface = ({ chatId, onNewChat }) => {
  const { t } = useTranslation();
  const { token } = useAuth();
  const { showToast } = useToast();
  const [message, setMessage] = useState('');
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);
  
  // Load chat messages when chatId changes
  useEffect(() => {
    if (chatId) {
      loadChatMessages();
    } else {
      setMessages([]);
    }
  }, [chatId]);
  
  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);
  
  // Auto-resize textarea based on content
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = textareaRef.current.scrollHeight + 'px';
    }
  }, [message]);
  
  const loadChatMessages = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/chat/history`, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });
      
      if (!response.ok) {
        throw new Error(`Failed to load chat messages: ${response.status} ${response.statusText}`);
      }
      
      const text = await response.text();
      if (!text) {
        throw new Error('Empty response from server');
      }
      
      try {
        const data = JSON.parse(text);
        const conversation = data.conversations?.find(conv => conv.id === chatId);
        if (conversation) {
          setMessages(conversation.messages);
        }
      } catch (parseError) {
        throw new Error(`Invalid JSON response: ${parseError.message}`);
      }
      
    } catch (err) {
      showToast({
        type: 'error',
        message: err.message
      });
      setMessages([]);
    }
  };
  
  const handleSend = async () => {
    if (message.trim() && !isLoading) {
      setIsLoading(true);
      
      try {
        // Send message
        const response = await fetch(`${API_BASE_URL}/ask`, {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            query: message
          })
        });
        
        if (!response.ok) {
          throw new Error(`Failed to send message: ${response.status} ${response.statusText}`);
        }
        
        const text = await response.text();
        if (!text) {
          throw new Error('Empty response from server');
        }
        
        try {
          const data = JSON.parse(text);
          
          // Update messages
          setMessages(prev => [
            ...prev,
            {
              role: 'user',
              content: message,
              timestamp: new Date().toISOString()
            },
            {
              role: 'assistant',
              content: data.response,
              timestamp: new Date().toISOString()
            }
          ]);
          
          // If this is a new conversation, notify parent
          if (data.conversation_id && (!chatId || chatId === 'new')) {
            onNewChat(data.conversation_id);
          }
          
          setMessage('');
          
          // Reset textarea height
          if (textareaRef.current) {
            textareaRef.current.style.height = 'auto';
          }
        } catch (parseError) {
          throw new Error(`Invalid JSON response: ${parseError.message}`);
        }
        
      } catch (error) {
        showToast({
          type: 'error',
          message: error.message
        });
      } finally {
        setIsLoading(false);
      }
    }
  };
  
  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };
  
  return (
    <div className="flex flex-col h-full relative overflow-hidden">
      <div className="flex-1 overflow-y-auto p-4 space-y-6 bg-[#ffffff22] rounded-[10px]">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-auto text-center p-6 text-gray-500 dark:text-gray-400">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-16 w-16 mb-4 text-gray-300 dark:text-gray-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
            </svg>
            <h3 className="text-lg font-medium text-gray-700 dark:text-gray-300">Ask questions about your documents</h3>
            <p className="mt-1 max-w-md">Start a conversation by uploading documents and asking questions about them.</p>
          </div>
        ) : (
          messages.map((msg, index) => (
            <ChatMessage key={index} message={msg} />
          ))
        )}
        
        {isLoading && (
          <div className="flex items-start space-x-3">
            <div className="w-8 h-8 rounded-full bg-primary flex items-center justify-center text-white">
              AI
            </div>
            <div className="bg-gray-100 dark:bg-gray-800 rounded-lg p-3 max-w-3xl">
              <div className="flex space-x-2">
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.4s' }}></div>
              </div>
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>
      
      <div className="p-4">
        <div className="flex space-x-3">
          <div className="flex-1 relative">
            <textarea
              ref={textareaRef}
              className="input w-full min-h-[76px] max-h-[200px] py-2.5 resize-none"
              placeholder={t('chat.placeholder')}
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              onKeyDown={handleKeyDown}
              rows={1}
            />
            <Button
              onClick={handleSend}
              disabled={!message.trim() || isLoading}
              className="absolute right-2 bottom-4 flex h-[40px] items-center gap-2 px-3"
            >
              <Send size={16} />
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
};

const ChatMessage = ({ message }) => {
  const [showSources, setShowSources] = useState(false);
  const isUser = message.role === 'user';
  
  const bubbleContainerClass = isUser
    ? 'relative max-w-[65%] ml-auto'
    : 'relative';

  return (
    <div
      className={`flex items-start space-x-3 ${isUser ? 'justify-end' : ''}`}
    >
      {!isUser && (
        <div className="w-8 h-8 rounded-full bg-primary flex items-center justify-center text-white flex-shrink-0">
          AI
        </div>
      )}

      <div className={bubbleContainerClass}>
        <div className={`rounded-lg p-3 ${
            isUser
              ? 'bg-primary text-white'
              : 'bg-gray-100 dark:bg-gray-800 text-gray-800 dark:text-gray-200'
          }`}
        >
          <div className="whitespace-pre-wrap">{message.content}</div>
          
          {!isUser && message.sources && message.sources.length > 0 && (
            <div className="mt-2 pt-2 border-t border-gray-200 dark:border-gray-700">
              <button
                onClick={() => setShowSources(!showSources)}
                className="flex items-center text-sm text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300"
              >
                <span className="mr-1">Sources</span>
                {showSources ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
              </button>
              
              {showSources && (
                <div className="mt-2 space-y-2">
                  {message.sources.map((source, idx) => (
                    <div key={idx} className="text-sm bg-gray-50 dark:bg-gray-900 p-2 rounded border border-gray-200 dark:border-gray-700">
                      <p className="text-xs text-gray-600 dark:text-gray-400">{source.text}</p>
                      <div className="mt-1 text-xs text-gray-500">
                        Similarity: {(source.similarity * 100).toFixed(1)}%
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
      
      {isUser && (
        <div className="w-8 h-8 rounded-full bg-gray-200 dark:bg-gray-700 flex items-center justify-center text-gray-700 dark:text-gray-300 flex-shrink-0">
          You
        </div>
      )}
    </div>
  );
};

export default ChatInterface;