// src/pages/ChatPage.jsx
import { useState, useEffect } from 'react';
import ChatInterface from '../components/features/ChatInterface';
import useStore from '../store/useStore';

const ChatPage = () => {
  const [isLoading, setIsLoading] = useState(false);
  const { currentConversation, setCurrentConversation, addMessage } = useStore();
  
  useEffect(() => {
    // Create a new conversation if none exists
    if (!currentConversation) {
      setCurrentConversation({
        id: Date.now(),
        title: 'New Conversation',
        messages: []
      });
    }
  }, [currentConversation, setCurrentConversation]);
  
  const handleSendMessage = async (message) => {
    // Add user message
    addMessage({
      id: Date.now(),
      role: 'user',
      content: message,
      timestamp: new Date().toISOString()
    });
    
    setIsLoading(true);
    
    // Simulate API call with a delay
    setTimeout(() => {
      // Simulate an AI response
      addMessage({
        id: Date.now() + 1,
        role: 'assistant',
        content: `This is a simulated response to: "${message}"`,
        timestamp: new Date().toISOString(),
        sources: [
          {
            title: 'Document 1',
            content: 'This is a relevant excerpt from Document 1 that helps answer the question.'
          },
          {
            title: 'Document 2',
            content: 'Here is another source from Document 2 with additional context and information.'
          }
        ]
      });
      
      setIsLoading(false);
    }, 1500);
  };
  
  return (
    <div className="flex flex-col h-full">
      <div className="flex-1 min-h-0">
        {currentConversation && (
          <ChatInterface
            messages={currentConversation.messages}
            onSendMessage={handleSendMessage}
            isLoading={isLoading}
          />
        )}
      </div>
      {/* Footer nằm ngoài ChatInterface, dính đáy page */}
      <footer className="border-t border-gray-200 dark:border-gray-700 text-center py-2 text-sm text-gray-500 dark:text-gray-400 mt-auto">
        © RAGvLangChain_demo v1
      </footer>
    </div>
  );
};

export default ChatPage;