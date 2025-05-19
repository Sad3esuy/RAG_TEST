// src/components/layout/Header.jsx
import { useNavigate } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { Menu, Bell, User, ChevronDown } from 'lucide-react';
import { useTheme } from '../../context/ThemeContext';
import Dropdown from '../common/Dropdown';

function Header({ onMenuClick }) {
  const { t } = useTranslation();
  const { isDark, toggleTheme } = useTheme();
  const navigate = useNavigate();
const Logo = () => (
  <div className="flex items-center">
    <img src="/RAG_logo.png" alt="logo" className="h-8 w-8 object-contain hidden md:block" />
    <span className="ml-2 text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-600 to-indigo-600">
      RAGvLangChain
    </span>
  </div>
);


  return (
    <header className="bg-white dark:bg-gray-800 shadow-sm border-b border-gray-200 dark:border-gray-700">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          <div className="flex items-center">
            <button
              onClick={onMenuClick}
              className="md:hidden p-2 rounded-md text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
            >
              <Menu size={24} />
            </button>
            <div className="flex-shrink-0 flex items-center">
              <Logo />
            </div>
          </div>
          
          <div className="flex items-center space-x-4">
            <button 
              onClick={toggleTheme}
              className="p-2 rounded-full hover:bg-gray-100 dark:hover:bg-gray-700"
            >
              {isDark ? (
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-yellow-400" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M10 2a1 1 0 011 1v1a1 1 0 11-2 0V3a1 1 0 011-1zm4 8a4 4 0 11-8 0 4 4 0 018 0zm-.464 4.95l.707.707a1 1 0 001.414-1.414l-.707-.707a1 1 0 00-1.414 1.414zm2.12-10.607a1 1 0 010 1.414l-.706.707a1 1 0 11-1.414-1.414l.707-.707a1 1 0 011.414 0zM17 11a1 1 0 100-2h-1a1 1 0 100 2h1zm-7 4a1 1 0 011 1v1a1 1 0 11-2 0v-1a1 1 0 011-1zM5.05 6.464A1 1 0 106.465 5.05l-.708-.707a1 1 0 00-1.414 1.414l.707.707zm1.414 8.486l-.707.707a1 1 0 01-1.414-1.414l.707-.707a1 1 0 011.414 1.414zM4 11a1 1 0 100-2H3a1 1 0 000 2h1z" clipRule="evenodd" />
                </svg>
              ) : (
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-gray-700" viewBox="0 0 20 20" fill="currentColor">
                  <path d="M17.293 13.293A8 8 0 016.707 2.707a8.001 8.001 0 1010.586 10.586z" />
                </svg>
              )}
            </button>
            
            <button className="p-2 rounded-full hover:bg-gray-100 dark:hover:bg-gray-700 relative">
              <Bell size={20} className="text-gray-600 dark:text-gray-300" />
              <span className="absolute top-1 right-1 bg-red-500 w-2 h-2 rounded-full"></span>
            </button>
            
            <Dropdown
              trigger={
                <div className="flex items-center cursor-pointer">
                  <div className="w-8 h-8 rounded-full bg-gray-200 dark:bg-gray-600 flex items-center justify-center">
                    <User size={18} className="text-gray-600 dark:text-gray-300" />
                    <div><a href=""></a></div>
                  </div>
                  <ChevronDown size={16} className="ml-1 text-gray-600 dark:text-gray-300" />
                </div>
              }
              items={[
                { label: t('settings.title'), onClick: () => navigate('/settings') },
                { label: 'logout', onClick: () => navigate('/auth') },
                // { label: 'register', onClick: () => navigate('/auth') },
                // { label: t('nav.history'), onClick: () => navigate('/history') },
                // { label: 'Logout', onClick: () => console.log('Logout') },
              ]}
            />
          </div>
        </div>
      </div>
    </header>
  );
}

export default Header;