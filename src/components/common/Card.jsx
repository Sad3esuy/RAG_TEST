// src/components/common/Card.jsx
const Card = ({ children, className = '', title, description, footer, isLoading = false }) => {
    return (
      <div className={`bg-white dark:bg-gray-800 rounded-lg shadow-md ${className}`}>
        {isLoading ? (
          <div className="animate-pulse space-y-4 p-6">
            <div className="h-6 bg-gray-200 dark:bg-gray-700 rounded w-3/4"></div>
            <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-1/2"></div>
            <div className="h-32 bg-gray-200 dark:bg-gray-700 rounded"></div>
          </div>
        ) : (
          <>
            {title && (
              <div className="p-6 border-b border-gray-100 dark:border-gray-700">
                <h3 className="text-lg font-medium">{title}</h3>
                {description && <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">{description}</p>}
              </div>
            )}
            
            <div className={title ? 'p-6' : ''}>{children}</div>
            
            {footer && (
              <div className="p-6 border-t border-gray-100 dark:border-gray-700">
                {footer}
              </div>
            )}
          </>
        )}
      </div>
    );
  };

export default Card;