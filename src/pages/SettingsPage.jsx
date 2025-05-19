import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useTheme } from '../context/ThemeContext';
import { useToast } from '../components/common/Toast';
import Button from '../components/common/Button';
import { Moon, Sun, Languages, Database, Trash2 } from 'lucide-react';

const SettingsPage = () => {
  const { t, i18n } = useTranslation();
  const { isDark, toggleTheme } = useTheme();
  const { showToast } = useToast();
  const [settings, setSettings] = useState({
    embeddingModel: 'text-embedding-ada-002',
    chunkSize: 1000,
    chunkOverlap: 200,
  });

  const handleLanguageChange = (language) => {
    i18n.changeLanguage(language);
    showToast({
      type: 'success',
      message: t('settings.languageChanged'),
    });
  };

  const handleSettingChange = (key, value) => {
    setSettings(prev => ({
      ...prev,
      [key]: value
    }));
  };

  const handleSaveSettings = () => {
    // In a real app, you would save these settings to your backend
    localStorage.setItem('ragSettings', JSON.stringify(settings));
    showToast({
      type: 'success',
      message: t('settings.saved'),
    });
  };

  const handleClearData = () => {
    if (window.confirm(t('settings.confirmClearData'))) {
      // In a real app, you would clear data from your backend
      localStorage.clear();
      showToast({
        type: 'success',
        message: t('settings.dataCleared'),
      });
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold">{t('settings.title')}</h1>
        <p className="text-gray-500 dark:text-gray-400 mt-1">
          {t('settings.description')}
        </p>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        {/* Theme Settings */}
        <div className="card">
          <h2 className="text-xl font-semibold mb-4">{t('settings.theme')}</h2>
          <div className="flex items-center space-x-4">
            <Button
              variant={!isDark ? 'primary' : 'outline'}
              onClick={() => toggleTheme()}
              className="flex items-center gap-2"
            >
              <Sun size={20} />
              {t('settings.light')}
            </Button>
            <Button
              variant={isDark ? 'primary' : 'outline'}
              onClick={() => toggleTheme()}
              className="flex items-center gap-2"
            >
              <Moon size={20} />
              {t('settings.dark')}
            </Button>
          </div>
        </div>

        {/* Language Settings */}
        <div className="card">
          <h2 className="text-xl font-semibold mb-4">{t('settings.language')}</h2>
          <div className="flex items-center space-x-4">
            <Button
              variant={i18n.language === 'en' ? 'primary' : 'outline'}
              onClick={() => handleLanguageChange('en')}
              className="flex items-center gap-2"
            >
              <Languages size={20} />
              English
            </Button>
            <Button
              variant={i18n.language === 'vi' ? 'primary' : 'outline'}
              onClick={() => handleLanguageChange('vi')}
              className="flex items-center gap-2 whitespace-nowrap"
            >
              <Languages size={20} />
              Tiếng Việt
            </Button>
          </div>
        </div>

        {/* RAG Settings */}
        <div className="card md:col-span-2">
          <h2 className="text-xl font-semibold mb-4">{t('settings.ragConfig')}</h2>
          <div className="grid gap-4 md:grid-cols-2">
            <div>
              <label className="block text-sm font-medium mb-1">
                {t('settings.embedding')}
              </label>
              <select
                className="input"
                value={settings.embeddingModel}
                onChange={(e) => handleSettingChange('embeddingModel', e.target.value)}
              >
                <option value="text-embedding-ada-002">text-embedding-ada-002</option>
                <option value="text-embedding-3-small">text-embedding-3-small</option>
                <option value="text-embedding-3-large">text-embedding-3-large</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium mb-1">
                {t('settings.chunkSize')}
              </label>
              <input
                type="number"
                className="input"
                value={settings.chunkSize}
                onChange={(e) => handleSettingChange('chunkSize', parseInt(e.target.value))}
                min="100"
                max="2000"
                step="100"
              />
            </div>
            <div>
              <label className="block text-sm font-medium mb-1">
                {t('settings.overlap')}
              </label>
              <input
                type="number"
                className="input"
                value={settings.chunkOverlap}
                onChange={(e) => handleSettingChange('chunkOverlap', parseInt(e.target.value))}
                min="0"
                max="500"
                step="50"
              />
            </div>
          </div>
        </div>

        {/* Danger Zone */}
        <div className="card md:col-span-2 border-2 border-error/20">
          <h2 className="text-xl font-semibold mb-4 text-error">{t('settings.dangerZone')}</h2>
          <div className="flex items-center justify-between">
            <div>
              <h3 className="font-medium">{t('settings.clearData')}</h3>
              <p className="text-sm text-gray-500 dark:text-gray-400">
                {t('settings.clearDataDescription')}
              </p>
            </div>
            <Button
              variant="danger"
              onClick={handleClearData}
              className="flex items-center gap-2 whitespace-nowrap"
            >
              <Trash2 size={20} />
              {t('settings.clearData')}
            </Button>
          </div>
        </div>
      </div>

      <div className="flex justify-end">
        <Button
          variant="primary"
          onClick={handleSaveSettings}
          className="flex items-center gap-2"
        >
          <Database size={20} />
          {t('common.save')}
        </Button>
      </div>
    </div>
  );
};

export default SettingsPage; 