import { createContext, useContext, useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useToast } from '../components/common/Toast';

const AuthContext = createContext(null);

export const AuthProvider = ({ children }) => {
  const [token, setToken] = useState(localStorage.getItem('token'));
  const [user, setUser] = useState(() => {
    // Try to get user data from localStorage on initial load
    const savedUser = localStorage.getItem('user');
    return savedUser ? JSON.parse(savedUser) : null;
  });
  const navigate = useNavigate();
  const { showToast } = useToast();
  
  useEffect(() => {
    console.log('Token changed:', token);
    // If token is removed, clear user data
    if (!token) {
      localStorage.removeItem('user');
      setUser(null);
    }
  }, [token]);
  
  const getRoleBasedRoute = (roleId) => {
    switch (roleId) {
      case 1: // Admin
        return '/dashboard';
      case 2: // User
        return '/chat';
      default:
        return '/chat'; // Default fallback
    }
  };
  
  const login = async (email, password) => {
    try {
      console.log('Attempting login...');
      const response = await fetch('/api/auth/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ email, password })
      });
      
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.message || 'Login failed');
      }
      
      const data = await response.json();
      console.log('Login response:', data);
      
      // Save token and user data
      localStorage.setItem('token', data.token);
      localStorage.setItem('user', JSON.stringify(data.user));
      setToken(data.token);
      setUser(data.user);
      
      showToast({
        type: 'success',
        message: 'Login successful'
      });
      
      // Navigate based on user role
      const targetRoute = getRoleBasedRoute(data.user.role_id);
      console.log('Navigating to:', targetRoute);
      navigate(targetRoute, { replace: true });
      
    } catch (error) {
      console.error('Login error:', error);
      showToast({
        type: 'error',
        message: error.message
      });
      throw error;
    }
  };
  
  const register = async (email, password, fullName) => {
    try {
      const response = await fetch('/api/auth/register', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ email, password, full_name: fullName })
      });
      
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.message || 'Registration failed');
      }
      
      const data = await response.json();
      
      // Save token and user data
      localStorage.setItem('token', data.token);
      localStorage.setItem('user', JSON.stringify(data.user));
      setToken(data.token);
      setUser(data.user);
      
      showToast({
        type: 'success',
        message: 'Registration successful'
      });
      
      // Navigate based on user role
      const targetRoute = getRoleBasedRoute(data.user.role_id);
      navigate(targetRoute, { replace: true });
      
    } catch (error) {
      showToast({
        type: 'error',
        message: error.message
      });
      throw error;
    }
  };
  
  const logout = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    setToken(null);
    setUser(null);
    navigate('/auth?mode=login');
  };
  
  const forgotPassword = async (email) => {
    try {
      const response = await fetch('/api/auth/forgot-password', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ email })
      });
      
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.message || 'Failed to send reset email');
      }
      
      showToast({
        type: 'success',
        message: 'Password reset email sent'
      });
      
    } catch (error) {
      showToast({
        type: 'error',
        message: error.message
      });
      throw error;
    }
  };
  
  const resetPassword = async (token, password) => {
    try {
      const response = await fetch('/api/auth/reset-password', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ token, password })
      });
      
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.message || 'Failed to reset password');
      }
      
      showToast({
        type: 'success',
        message: 'Password reset successful'
      });
      
      navigate('/login');
      
    } catch (error) {
      showToast({
        type: 'error',
        message: error.message
      });
      throw error;
    }
  };
  
  const value = {
    token,
    user,
    login,
    register,
    logout,
    forgotPassword,
    resetPassword
  };
  
  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}; 