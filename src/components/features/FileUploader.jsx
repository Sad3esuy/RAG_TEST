// src/components/features/FileUploader.jsx
import { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { useTranslation } from 'react-i18next';
import { motion } from 'framer-motion';
import { Upload, File, X } from 'lucide-react';
import Button from '../common/Button';
import { useToast } from '../common/Toast';
import { useAuth } from '../../context/AuthContext';

const API_BASE_URL = 'http://127.0.0.1:8000';

const FileUploader = ({ onUpload }) => {
  const { t } = useTranslation();
  const { showToast } = useToast();

  const onDrop = useCallback(async (acceptedFiles) => {
    // Chỉ cho phép PDF
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
        const formData = new FormData();
        formData.append('file', fileData.file);

        const response = await fetch(`${API_BASE_URL}/upload_pdf`, {
          method: 'POST',
          body: formData
        });

        const text = await response.text();
        let data;
        try {
          data = JSON.parse(text);
        } catch {
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
  }, [onUpload, showToast]);

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
          <motion.div
            key={file.id}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
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
              
              {file.status === 'uploading' && (
                <div className="mt-1 w-full bg-gray-200 dark:bg-gray-700 rounded-full h-1.5">
                  <div 
                    className="bg-primary h-1.5 rounded-full transition-all duration-300" 
                    style={{ width: `${file.progress}%` }}
                  ></div>
                </div>
              )}
              
              {file.status === 'error' && (
                <p className="text-xs text-red-500 mt-1">
                  Upload failed. Please try again.
                </p>
              )}
            </div>
            
            <div className="ml-4">
              {file.status === 'completed' ? (
                <button
                  onClick={() => handleRemove(file.id, file.documentId)}
                  className="p-1 rounded-full hover:bg-gray-200 dark:hover:bg-gray-700"
                >
                  <X size={16} className="text-gray-500 dark:text-gray-400" />
                </button>
              ) : file.status === 'uploading' ? (
                <span className="text-xs text-gray-500 dark:text-gray-400">
                  {file.progress}%
                </span>
              ) : (
                <button
                  onClick={() => handleRemove(file.id)}
                  className="p-1 rounded-full hover:bg-gray-200 dark:hover:bg-gray-700"
                >
                  <X size={16} className="text-gray-500 dark:text-gray-400" />
                </button>
              )}
            </div>
          </motion.div>
        ))}
      </div>
    </div>
  );
};

export default FileUploader;