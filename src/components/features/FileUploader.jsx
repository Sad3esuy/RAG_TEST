// src/components/features/FileUploader.jsx
import { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { useTranslation } from 'react-i18next';
import { Upload, File, X } from 'lucide-react';
import Button from '../common/Button';
import { useToast } from '../common/Toast';
import { useAuth } from '../../context/AuthContext';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const FileUploader = ({ onUpload }) => {
  const { t } = useTranslation();
  const { showToast } = useToast();
  const { token } = useAuth();

  const onDrop = useCallback(async (acceptedFiles) => {
    // Only allow PDFs
    const validFiles = acceptedFiles.filter(file => file.type === 'application/pdf');

    if (validFiles.length !== acceptedFiles.length) {
      showToast({
        type: 'error',
        message: 'Only PDF files are supported.'
      });
    }
    if (validFiles.length === 0) return;

    // Process files
    const processedFiles = validFiles.map(file => ({
      id: Date.now() + Math.random().toString(36).substr(2, 9),
      name: file.name,
      size: file.size,
      type: file.type,
      progress: 0,
      status: 'uploading',
      file
    }));

    // Notify parent component
    onUpload(processedFiles);

    // Upload files to FastAPI backend
    for (const fileData of processedFiles) {
      try {
        console.log(`Uploading file: ${fileData.name}`);
        const formData = new FormData();
        formData.append('file', fileData.file);

        const response = await fetch(`${API_BASE_URL}/upload_pdf`, {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${token}`
          },
          body: formData
        });

        console.log('Upload response status:', response.status);
        const text = await response.text();
        console.log('Upload response text:', text);

        let data;
        try {
          data = JSON.parse(text);
        } catch (e) {
          console.error('Failed to parse response:', e);
          throw new Error('Invalid response from server: ' + text);
        }

        if (!response.ok) {
          throw new Error(data.detail || 'Upload failed');
        }

        onUpload((prev) =>
          prev.map(f =>
            f.id === fileData.id
              ? {
                  ...f,
                  status: 'completed',
                  progress: 100
                }
              : f
          )
        );

        showToast({
          type: 'success',
          message: data.message || `${fileData.name} uploaded successfully!`
        });
      } catch (err) {
        console.error('Upload error:', err);
        onUpload((prev) =>
          prev.map(f =>
            f.id === fileData.id
              ? {
                  ...f,
                  status: 'error',
                  progress: 0
                }
              : f
          )
        );
        showToast({
          type: 'error',
          message: `Failed to upload ${fileData.name}: ${err.message}`
        });
      }
    }
  }, [onUpload, showToast, token]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({ onDrop });

  return (
    <div className="w-full">
      <div
        {...getRootProps()}
        className={`border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition-colors
          ${isDragActive
            ? 'border-primary bg-primary/5'
            : 'border-gray-300 dark:border-gray-700 hover:border-primary dark:hover:border-primary'
          }`}
      >
        <input {...getInputProps()} accept="application/pdf" />
        <Upload className="mx-auto h-12 w-12 text-gray-400" />
        <p className="mt-2 text-sm font-medium text-gray-900 dark:text-gray-300">
          {t('import.dropzone')}
        </p>
        <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
          {t('import.supported', 'Only PDF files are supported.')}
        </p>
      </div>
    </div>
  );
};

export const FileUploadList = ({ files, onRemove }) => {
  const { token } = useAuth();
  const { showToast } = useToast();
  
  const handleRemove = async (fileId, documentId) => {
    if (!documentId) {
      onRemove(fileId);
      return;
    }
    
    try {
      const response = await fetch(`/api/files/documents/${documentId}`, {
        method: 'DELETE',
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });
      
      if (!response.ok) {
        throw new Error('Failed to delete document');
      }
      
      onRemove(fileId);
      
      showToast({
        type: 'success',
        message: 'Document deleted successfully'
      });
      
    } catch (err) {
      showToast({
        type: 'error',
        message: `Failed to delete document: ${err.message}`
      });
    }
  };
  
  const getFileIcon = (fileType) => {
    if (fileType.includes('pdf')) {
      return "📄";
    } else if (fileType.includes('word')) {
      return "📝";
    } else if (fileType.includes('text')) {
      return "📑";
    }
    return "📁";
  };
  
  const formatFileSize = (bytes) => {
    if (bytes < 1024) return bytes + ' B';
    else if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
    else return (bytes / 1048576).toFixed(1) + ' MB';
  };
  
  if (files.length === 0) return null;
  
  return (
    <div className="mt-6 space-y-4">
      <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300">Uploaded files</h3>
      
      <div className="space-y-3">
        {files.map((file) => (
          <div
            key={file.id}
            className="flex items-center p-3 bg-gray-50 dark:bg-gray-800 rounded-lg"
          >
            <div className="mr-3 text-xl">{getFileIcon(file.type)}</div>
            
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium text-gray-900 dark:text-gray-100 truncate">
                {file.name}
              </p>
              <p className="text-xs text-gray-500 dark:text-gray-400">
                {formatFileSize(file.size)}
              </p>
            </div>
            
            <div className="ml-4 flex items-center">
              {file.status === 'uploading' && (
                <div className="w-16 h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-primary transition-all duration-300"
                    style={{ width: `${file.progress}%` }}
                  />
                </div>
              )}
              
              <button
                onClick={() => handleRemove(file.id, file.documentId)}
                className="ml-4 p-1 text-gray-400 hover:text-gray-500 dark:hover:text-gray-300"
              >
                <X className="h-5 w-5" />
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default FileUploader;