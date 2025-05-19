// src/pages/ImportPage.jsx
import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import FileUploader, { FileUploadList } from '../components/features/FileUploader';
import Card from '../components/common/Card';
import { useToast } from '../components/common/Toast';

const ImportPage = () => {
  const { t } = useTranslation();
  const { showToast } = useToast();
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);

  const handleUpload = async (files) => {
    // Nếu files là function (callback), bỏ qua
    if (typeof files === 'function') return;

    setUploadedFiles(files);
    setIsProcessing(true);

    try {
      if (Array.isArray(files)) {
        const allCompleted = files.every(file => file.status === 'completed');
        const anyErrors = files.some(file => file.status === 'error');

        if (allCompleted) {
          showToast({
            type: 'success',
            message: t('import.success', 'All files uploaded successfully')
          });
        } else if (anyErrors) {
          showToast({
            type: 'error',
            message: t('import.error', 'Some files failed to upload')
          });
        }
      }
    } catch (err) {
      showToast({
        type: 'error',
        message: t('import.error', 'Failed to process files: ') + err.message
      });
    } finally {
      setIsProcessing(false);
    }
  };

  const handleRemoveFile = (fileId) => {
    setUploadedFiles(prev => prev.filter(file => file.id !== fileId));
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold">{t('import.title')}</h1>
        <p className="text-gray-500 dark:text-gray-400 mt-1">
          {t('import.description', 'Upload documents to build a knowledge base. Supported formats: PDF, DOCX, TXT, MD')}
        </p>
      </div>

      <Card>
        <div className="space-y-4">
          <FileUploader onUpload={handleUpload} />
          <FileUploadList 
            files={uploadedFiles} 
            onRemove={handleRemoveFile}
          />
          
          {isProcessing && (
            <div className="mt-4 p-3 bg-gray-100 dark:bg-gray-800 rounded-md text-sm text-gray-700 dark:text-gray-300">
              <p>{t('import.processing', 'Processing files...')}</p>
            </div>
          )}
        </div>
      </Card>
    </div>
  );
};

export default ImportPage;