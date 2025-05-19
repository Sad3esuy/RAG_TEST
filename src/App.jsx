import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider } from './context/AuthContext';
import { ThemeProvider } from './context/ThemeContext';
import { ToastProvider } from './components/common/Toast';
import ProtectedRoute from './components/auth/ProtectedRoute';
import Layout from './components/layout/Layout';
import AuthPage from './pages/auth/auth';
import Dashboard from './pages/Dashboard';
import ImportPage from './pages/ImportPage';
import ChatPage from './pages/ChatPage';
import HistoryPage from './pages/HistoryPage';
import SettingsPage from './pages/SettingsPage';
import './i18n';

function App() {
  return (
    <Router>
      <ThemeProvider>
        <ToastProvider>
          <AuthProvider>
            <Routes>
              {/* Auth routes */}
              <Route path="/auth" element={<AuthPage />} />
              
              {/* Protected routes with layout */}
              <Route element={<Layout />}>
                {/* Admin routes */}
                <Route path="/dashboard" element={
                  <ProtectedRoute requiredRole={1}>
                    <Dashboard />
                  </ProtectedRoute>
                } />
                <Route path="/import" element={
                  <ProtectedRoute requiredRole={1}>
                    <ImportPage />
                  </ProtectedRoute>
                } />
                
                {/* User routes */}
                <Route path="/chat" element={
                  <ProtectedRoute requiredRole={2}>
                    <ChatPage />
                  </ProtectedRoute>
                } />
                <Route path="/history" element={
                  <ProtectedRoute requiredRole={2}>
                    <HistoryPage />
                  </ProtectedRoute>
                } />
                
                {/* Common routes */}
                <Route path="/settings" element={
                  <ProtectedRoute>
                    <SettingsPage />
                  </ProtectedRoute>
                } />
                
                {/* Redirect root based on role */}
                <Route path="/" element={
                  <ProtectedRoute>
                    {({ user }) => (
                      <Navigate to={user?.role_id === 1 ? '/dashboard' : '/chat'} replace />
                    )}
                  </ProtectedRoute>
                } />
              </Route>
            </Routes>
          </AuthProvider>
        </ToastProvider>
      </ThemeProvider>
    </Router>
  );
}

export default App;
