// src/store/useStore.js
import { create } from 'zustand';

const useStore = create((set) => ({
  // Dashboard stats
  stats: {
    documentCount: 0,
    queryCount: 0,
    avgResponseTime: 0,
  },
  
  // Documents
  documents: [],
  
  // Chat history
  conversations: [],
  currentConversation: null,
  
  // Actions
  setStats: (stats) => set({ stats }),
  setDocuments: (documents) => set({ documents }),
  addDocument: (document) => set((state) => ({ 
    documents: [...state.documents, document],
    stats: { 
      ...state.stats, 
      documentCount: state.stats.documentCount + 1 
    }
  })),
  removeDocument: (id) => set((state) => ({ 
    documents: state.documents.filter(doc => doc.id !== id),
    stats: { 
      ...state.stats, 
      documentCount: state.stats.documentCount - 1 
    }
  })),
  
  setConversations: (conversations) => set({ conversations }),
  setCurrentConversation: (conversation) => set({ currentConversation: conversation }),
  addMessage: (message) => set((state) => {
    if (!state.currentConversation) return state;
    
    const updatedConversation = {
      ...state.currentConversation,
      messages: [...state.currentConversation.messages, message]
    };
    
    const updatedConversations = state.conversations.map(conv => 
      conv.id === updatedConversation.id ? updatedConversation : conv
    );
    
    if (!updatedConversations.some(conv => conv.id === updatedConversation.id)) {
      updatedConversations.push(updatedConversation);
    }
    
    return {
      currentConversation: updatedConversation,
      conversations: updatedConversations,
      stats: {
        ...state.stats,
        queryCount: message.role === 'user' ? state.stats.queryCount + 1 : state.stats.queryCount
      }
    };
  }),
}));

export default useStore;