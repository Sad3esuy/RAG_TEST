// src/pages/HistoryPage.jsx
import { useState, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { format } from 'date-fns';
import { Search, Trash2, MessageSquare } from 'lucide-react';
import Card from '../components/common/Card';
import Input from '../components/common/Input';
import Button from '../components/common/Button';
import useStore from '../store/useStore';
import { useNavigate } from 'react-router-dom';

const HistoryPage = () => {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const [searchTerm, setSearchTerm] = useState('');
  const { conversations, setCurrentConversation } = useStore();
  const [filteredHistory, setFilteredHistory] = useState([]);
  
  useEffect(() => {
    // For demo purposes, let's generate some sample history
    if (conversations.length === 0) {
      const sampleConversations = Array.from({ length: 10 }, (_, i) => ({
        id: i + 1,
        title: `Conversation ${i + 1}`,
        lastMessage: `This is a sample question about document ${i}?`,
        timestamp: new Date(Date.now() - i * 86400000).toISOString(),
        messages: [
          {
            id: `q-${i}`,
            role: 'user',
            content: `This is a sample question about document ${i}?`,
            timestamp: new Date(Date.now() - i * 86400000).toISOString()
          },
          {
            id: `a-${i}`,
            role: 'assistant',
            content: `This is a sample answer to the question about document ${i}.`,
            timestamp: new Date(Date.now() - i * 86400000 + 1000).toISOString()
          }
        ]
      }));
      
      // In a real app, you would update the store with these conversations
      // For now, we'll just use them locally
      setFilteredHistory(sampleConversations);
    } else {
      setFilteredHistory(conversations);
    }
  }, [conversations]);
  
  useEffect(() => {
    // Filter conversations based on search term
    if (searchTerm.trim() === '') {
      setFilteredHistory(conversations);
    } else {
      const filtered = conversations.filter(conv => 
        conv.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
        conv.lastMessage.toLowerCase().includes(searchTerm.toLowerCase())
      );
      setFilteredHistory(filtered);
    }
  }, [searchTerm, conversations]);
  
  const handleConversationClick = (conversation) => {
    setCurrentConversation(conversation);
    navigate('/chat');
  };
  
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold">{t('history.title')}</h1>
        <p className="text-gray-500 dark:text-gray-400 mt-1">
          {t('history.description')}
        </p>
      </div>
      
      <Card>
        <div className="mb-4">
          <Input
            placeholder={t('history.searchPlaceholder')}
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            icon={<Search size={18} />}
          />
        </div>
        
        <div className="space-y-2">
          {filteredHistory.length === 0 ? (
            <p className="text-center text-gray-500 py-4">{t('history.noResults')}</p>
          ) : (
            filteredHistory.map((conversation) => (
              <div 
                key={conversation.id}
                className="flex items-center justify-between p-3 rounded-md hover:bg-gray-100 dark:hover:bg-gray-700 cursor-pointer"
                onClick={() => handleConversationClick(conversation)}
              >
                <div className="flex items-center">
                  <MessageSquare size={18} className="mr-3 text-gray-500" />
                  <div>
                    <h3 className="font-medium">{conversation.title}</h3>
                    <p className="text-sm text-gray-500">{conversation.lastMessage}</p>
                  </div>
                </div>
                <div className="flex items-center">
                  <span className="text-xs text-gray-500 mr-3">
                    {format(new Date(conversation.timestamp), 'MMM d, yyyy')}
                  </span>
                  <Button
                    variant="ghost"
                    size="sm"
                    icon={<Trash2 size={16} />}
                    onClick={(e) => {
                      e.stopPropagation();
                      // Handle delete conversation
                    }}
                  />
                </div>
              </div>
            ))
          )}
        </div>
      </Card>
    </div>
  );
};

export default HistoryPage;