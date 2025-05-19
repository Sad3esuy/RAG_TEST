import { Navigate, useLocation, Outlet } from 'react-router-dom';
import { useAuth } from '../../context/AuthContext';

const ProtectedRoute = ({ children, requiredRole }) => {
  const { token, user } = useAuth();
  const location = useLocation();

  console.log('ProtectedRoute - Token:', token);
  console.log('ProtectedRoute - User:', user);
  console.log('ProtectedRoute - Location:', location);
  console.log('ProtectedRoute - Required Role:', requiredRole);

  if (!token) {
    console.log('ProtectedRoute - No token, redirecting to login');
    return <Navigate to="/auth?mode=login" state={{ from: location }} replace />;
  }

  // Check role if required
  if (requiredRole && user && user.role_id !== requiredRole) {
    console.log('ProtectedRoute - Role mismatch, redirecting to appropriate route');
    // Redirect based on user's role
    const targetRoute = user.role_id === 1 ? '/dashboard' : '/chat';
    return <Navigate to={targetRoute} replace />;
  }

  console.log('ProtectedRoute - Access granted, rendering protected content');
  return children ? children : <Outlet />;
};

export default ProtectedRoute;
