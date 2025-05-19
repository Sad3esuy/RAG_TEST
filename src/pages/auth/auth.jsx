import { useState, useEffect } from 'react';
import { Eye, EyeOff, Mail, Lock, User, AlertCircle, CheckCircle2, ArrowRight, KeyRound, Loader2 } from 'lucide-react';
import { useAuth } from '../../context/AuthContext';

// Logo Component
const Logo = () => (
  <div className="flex items-center">
      <img
        src="/RAG_logo.png" // Nếu ảnh nằm trong thư mục "public"
        alt="logo"
        className="h-8 w-8 rounded-full"
      />
    <span className="ml-2 text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-600 to-indigo-600">RAGvLangChain</span>
  </div>
);

const Input = ({ label, type, value, onChange, error, placeholder, icon: Icon }) => {
  const [show, setShow] = useState(false);
  const isPassword = type === 'password';

  return (
    <div className="mb-4 relative">
      <label className="block text-sm font-medium text-gray-700 mb-1">{label}</label>
      <div className="relative">
        {Icon && (
          <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-gray-500">
            <Icon size={16} />
          </div>
        )}
        <input
          type={isPassword ? (show ? 'text' : 'password') : type}
          value={value}
          onChange={onChange}
          placeholder={placeholder}
          className={`w-full ${Icon ? 'pl-10' : 'pl-3'} pr-10 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 transition-all ${
            error ? 'border-red-500' : 'border-gray-300'
          }`}
        />
        {isPassword && (
          <button
            type="button"
            onClick={() => setShow(s => !s)}
            className="absolute inset-y-0 right-3 flex items-center p-1 text-gray-500 hover:text-gray-700 transition-colors"
            tabIndex={-1}
          >
            {show ? <EyeOff size={16} /> : <Eye size={16} />}
          </button>
        )}
      </div>
      {error && (
        <div className="mt-1 flex items-center text-sm text-red-500">
          <AlertCircle size={14} className="mr-1" />
          {error}
        </div>
      )}
    </div>
  );
};

const Button = ({ children, onClick, type = "button", disabled, variant = "primary", className = "", icon: Icon }) => {
  const baseClasses = "py-2 px-4 rounded-lg font-medium transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-offset-2 flex items-center justify-center";
  const variantClasses = {
    primary: "bg-blue-600 hover:bg-blue-700 text-white focus:ring-blue-500 shadow-sm hover:shadow",
    secondary: "bg-gray-100 hover:bg-gray-200 text-gray-800 focus:ring-gray-500",
    danger: "bg-red-600 hover:bg-red-700 text-white focus:ring-red-500",
    link: "bg-transparent hover:underline text-blue-600 hover:text-blue-800 p-0"
  };
  
  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      className={`${baseClasses} ${variantClasses[variant]} ${disabled ? 'opacity-70 cursor-not-allowed' : ''} ${className}`}
    >
      {disabled && <Loader2 size={16} className="animate-spin mr-2" />}
      {!disabled && Icon && <Icon size={16} className="mr-2" />}
      {children}
    </button>
  );
};

const SocialButton = ({ children, onClick, provider }) => {
  const icons = {
    google: (
      <svg className="w-5 h-5 mr-2" viewBox="0 0 24 24">
        <path
          fill="currentColor"
          d="M12.545,10.239v3.821h5.445c-0.712,2.315-2.647,3.972-5.445,3.972c-3.332,0-6.033-2.701-6.033-6.032s2.701-6.032,6.033-6.032c1.498,0,2.866,0.549,3.921,1.453l2.814-2.814C17.503,2.988,15.139,2,12.545,2C7.021,2,2.543,6.477,2.543,12s4.478,10,10.002,10c8.396,0,10.249-7.85,9.426-11.748L12.545,10.239z"
        />
      </svg>
    ),
    facebook: (
      <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="currentColor" className="bi bi-facebook" viewBox="0 0 19 19">
        <path d="M16 8.049c0-4.446-3.582-8.05-8-8.05C3.58 0-.002 3.603-.002 8.05c0 4.017 2.926 7.347 6.75 7.951v-5.625h-2.03V8.05H6.75V6.275c0-2.017 1.195-3.131 3.022-3.131.876 0 1.791.157 1.791.157v1.98h-1.009c-.993 0-1.303.621-1.303 1.258v1.51h2.218l-.354 2.326H9.25V16c3.824-.604 6.75-3.934 6.75-7.951"/>
      </svg>
    ),
    github: (
      <svg className="w-5 h-5 mr-2" viewBox="0 0 24 24">
          <path fill="currentColor" d="M10.9,2.1c-4.6,0.5-8.3,4.2-8.8,8.7c-0.6,5,2.5,9.3,6.9,10.7v-2.3c0,0-0.4,0.1-0.9,0.1c-1.4,0-2-1.2-2.1-1.9 c-0.1-0.4-0.3-0.7-0.6-1C5.1,16.3,5,16.3,5,16.2C5,16,5.3,16,5.4,16c0.6,0,1.1,0.7,1.3,1c0.5,0.8,1.1,1,1.4,1c0.4,0,0.7-0.1,0.9-0.2 c0.1-0.7,0.4-1.4,1-1.8c-2.3-0.5-4-1.8-4-4c0-1.1,0.5-2.2,1.2-3C7.1,8.8,7,8.3,7,7.6C7,7.2,7,6.6,7.3,6c0,0,1.4,0,2.8,1.3 C10.6,7.1,11.3,7,12,7s1.4,0.1,2,0.3C15.3,6,16.8,6,16.8,6C17,6.6,17,7.2,17,7.6c0,0.8-0.1,1.2-0.2,1.4c0.7,0.8,1.2,1.8,1.2,3 c0,2.2-1.7,3.5-4,4c0.6,0.5,1,1.4,1,2.3v3.3c4.1-1.3,7-5.1,7-9.5C22,6.1,16.9,1.4,10.9,2.1z"></path>
      </svg>
    )
  };

  return (
    <button
      type="button"
      onClick={onClick}
      className="w-full flex justify-center items-center py-2 px-4 border border-gray-300 rounded-lg shadow-sm bg-white hover:bg-gray-50 transition-colors text-sm font-medium text-gray-700"
    >
      {icons[provider]}
      {children}
    </button>
  );
};

// Component thông báo
const Alert = ({ type, message, onClose }) => {
  if (!message) return null;
  
  const icons = {
    error: <AlertCircle size={20} />,
    success: <CheckCircle2 size={20} />
  };
  
  const styles = {
    error: "bg-red-50 border-red-500 text-red-700",
    success: "bg-green-50 border-green-500 text-green-700"
  };
  
  return (
    <div className={`mb-4 border-l-4 p-4 ${styles[type]}`}>
      <div className="flex items-start">
        <div className={`flex-shrink-0 ${type === 'error' ? 'text-red-500' : 'text-green-500'}`}>
          {icons[type]}
        </div>
        <div className="ml-3 flex-1">
          <p className="text-sm">{message}</p>
        </div>
        {onClose && (
          <button 
            onClick={onClose}
            className={`ml-auto ${type === 'error' ? 'text-red-500' : 'text-green-500'} hover:opacity-70`}
          >
            <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        )}
      </div>
    </div>
  );
};

// Form chuyển đổi với hiệu ứng mượt mà và responsive
const AuthPage = () => {
  const { login, register, forgotPassword } = useAuth();
  const [activeForm, setActiveForm] = useState('login'); // 'login', 'register', 'forgot'
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);

  // Login state
  const [loginEmail, setLoginEmail] = useState('');
  const [loginPassword, setLoginPassword] = useState('');
  const [loginErrors, setLoginErrors] = useState({});
  const [rememberMe, setRememberMe] = useState(false);

  // Register state
  const [registerName, setRegisterName] = useState('');
  const [registerEmail, setRegisterEmail] = useState('');
  const [registerPassword, setRegisterPassword] = useState('');
  const [registerConfirmPassword, setRegisterConfirmPassword] = useState('');
  const [acceptedTerms, setAcceptedTerms] = useState(false);
  const [registerErrors, setRegisterErrors] = useState({});

  // Forgot state
  const [forgotEmail, setForgotEmail] = useState('');
  const [forgotErrors, setForgotErrors] = useState({});

  // Password strength indicator
  const [passwordStrength, setPasswordStrength] = useState(0);

  useEffect(() => {
    setError(null);
    setSuccess(null);
    setLoginErrors({});
    setRegisterErrors({});
    setForgotErrors({});
  }, [activeForm]);

  // Validate email
  const validateEmail = email => /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
  
  // Password strength checker
  const checkPasswordStrength = password => {
    if (!password) return 0;
    
    let score = 0;
    if (password.length >= 8) score += 1;
    if (password.length >= 12) score += 1;
    if (/[A-Z]/.test(password)) score += 1;
    if (/[0-9]/.test(password)) score += 1;
    if (/[!@#$%^&*(),.?":{}|<>]/.test(password)) score += 1;
    
    return score;
  };

  // Update password strength when password changes
  useEffect(() => {
    setPasswordStrength(checkPasswordStrength(registerPassword));
  }, [registerPassword]);

  const validatePasswordStrength = pwd => {
    if (pwd.length < 8) return 'Ít nhất 8 ký tự';
    if (!/[A-Z]/.test(pwd)) return 'Ít nhất 1 chữ hoa';
    if (!/[0-9]/.test(pwd)) return 'Ít nhất 1 chữ số';
    if (!/[!@#$%^&*(),.?":{}|<>]/.test(pwd)) return 'Ít nhất 1 ký tự đặc biệt';
    return null;
  };

  const handleLogin = async e => {
    if (e) e.preventDefault();
    const errs = {};
    if (!loginEmail) errs.email = 'Email is required';
    else if (!validateEmail(loginEmail)) errs.email = 'Invalid email format';
    if (!loginPassword) errs.password = 'Password is required';
    setLoginErrors(errs);
    if (Object.keys(errs).length) return;

    setIsLoading(true);
    setError(null);
    try {
      await login(loginEmail, loginPassword);
      setSuccess('Login successful!');
    } catch (error) {
      setError(error.message || 'Login failed. Please check your credentials.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleRegister = async e => {
    if (e) e.preventDefault();
    const errs = {};
    if (!registerName) errs.name = 'Full name is required';
    if (!registerEmail) errs.email = 'Email is required';
    else if (!validateEmail(registerEmail)) errs.email = 'Invalid email format';
    const pwdErr = validatePasswordStrength(registerPassword);
    if (!registerPassword) errs.password = 'Password is required';
    else if (pwdErr) errs.password = pwdErr;
    if (!registerConfirmPassword) errs.confirm = 'Password confirmation is required';
    else if (registerConfirmPassword !== registerPassword) errs.confirm = 'Passwords do not match';
    if (!acceptedTerms) errs.terms = 'You must accept the Terms & Conditions';

    setRegisterErrors(errs);
    if (Object.keys(errs).length) return;

    setIsLoading(true);
    setError(null);
    try {
      await register(registerEmail, registerPassword, registerName);
      setSuccess('Registration successful!');
    } catch (error) {
      setError(error.message || 'Registration failed. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleForgotPassword = async e => {
    if (e) e.preventDefault();
    const errs = {};
    if (!forgotEmail) errs.email = 'Email is required';
    else if (!validateEmail(forgotEmail)) errs.email = 'Invalid email format';
    setForgotErrors(errs);
    if (Object.keys(errs).length) return;

    setIsLoading(true);
    setError(null);
    try {
      await forgotPassword(forgotEmail);
      setSuccess('Password reset instructions have been sent to your email.');
    } catch (error) {
      setError(error.message || 'Failed to send reset email. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const renderPasswordStrength = () => {
    const labels = ['Yếu', 'Trung bình', 'Khá', 'Mạnh', 'Rất mạnh'];
    const colors = ['bg-red-500', 'bg-orange-500', 'bg-yellow-500', 'bg-green-400', 'bg-green-600'];
    
    return (
      <div className="mt-1 mb-3">
        <div className="flex justify-between mb-1">
          <div className="text-xs text-gray-600">Độ mạnh mật khẩu: {passwordStrength > 0 ? labels[passwordStrength - 1] : 'Rất yếu'}</div>
        </div>
        <div className="w-full h-1 bg-gray-200 rounded-full overflow-hidden flex">
          {[...Array(5)].map((_, i) => (
            <div
              key={i}
              className={`h-full ${i < passwordStrength ? colors[i] : 'bg-gray-200'} ${i > 0 ? 'ml-1' : ''}`}
              style={{ width: '20%' }}
            ></div>
          ))}
        </div>
      </div>
    );
  };

  const renderSocialButtons = () => (
    <>
      <div className="relative mt-6">
        <div className="absolute inset-0 flex items-center">
          <div className="w-full border-t border-gray-300"></div>
        </div>
        <div className="relative flex justify-center text-sm">
          <span className="px-2 bg-white text-gray-500">Hoặc tiếp tục với</span>
        </div>
      </div>
      <div className="mt-6 grid grid-cols-1 sm:grid-cols-3 gap-3">
        <SocialButton provider="google">Google</SocialButton>
        <SocialButton provider="facebook">Facebook</SocialButton>
        <SocialButton provider="github">GitHub</SocialButton>
      </div>
    </>
  );

  const renderLoginForm = () => (
    <form onSubmit={handleLogin} className="space-y-5 animate-fadeIn">
      <div className="text-center mb-6">
        <h2 className="text-2xl font-bold text-gray-800">Đăng nhập</h2>
        <p className="text-gray-600 mt-1">Chào mừng bạn quay trở lại!</p>
      </div>
      <Input
        label="Email"
        type="email"
        value={loginEmail}
        onChange={e => setLoginEmail(e.target.value)}
        placeholder="email@example.com"
        error={loginErrors.email}
        icon={Mail}
      />
      <Input
        label="Mật khẩu"
        type="password"
        value={loginPassword}
        onChange={e => setLoginPassword(e.target.value)}
        placeholder="••••••••"
        error={loginErrors.password}
        icon={Lock}
      />
      <div className="flex flex-wrap justify-between items-center text-sm">
        <div className="flex items-center mb-2 sm:mb-0">
          <input
            id="remember"
            type="checkbox"
            checked={rememberMe}
            onChange={e => setRememberMe(e.target.checked)}
            className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
          />
          <label htmlFor="remember" className="ml-2 block text-gray-700">Ghi nhớ đăng nhập</label>
        </div>
        <button
          type="button"
          onClick={() => setActiveForm('forgot')}
          className="text-blue-600 hover:text-blue-800 hover:underline focus:outline-none"
        >
          Quên mật khẩu?
        </button>
      </div>
      <Button type="submit" disabled={isLoading} className="w-full">
        {isLoading ? 'Đang xử lý...' : 'Đăng nhập'}
      </Button>
      <div className="mt-6 text-center">
        <p className="text-gray-600">
          Chưa có tài khoản?{' '}
          <button
            type="button"
            onClick={() => setActiveForm('register')}
            className="text-blue-600 hover:text-blue-800 hover:underline font-medium focus:outline-none"
          >
            Đăng ký ngay
          </button>
        </p>
      </div>
      {renderSocialButtons()}
    </form>
  );

  const renderRegisterForm = () => (
    <form onSubmit={handleRegister} className="space-y-4 animate-fadeIn">
      <div className="text-center mb-6">
        <h2 className="text-2xl font-bold text-gray-800">Đăng ký tài khoản</h2>
        <p className="text-gray-600 mt-1">Tạo tài khoản mới chỉ trong vài bước đơn giản</p>
      </div>
      <Input
        label="Họ và tên"
        type="text"
        value={registerName}
        onChange={e => setRegisterName(e.target.value)}
        placeholder="Nguyễn Văn A"
        error={registerErrors.name}
        icon={User}
      />
      <Input
        label="Email"
        type="email"
        value={registerEmail}
        onChange={e => setRegisterEmail(e.target.value)}
        placeholder="email@example.com"
        error={registerErrors.email}
        icon={Mail}
      />
      <Input
        label="Mật khẩu"
        type="password"
        value={registerPassword}
        onChange={e => setRegisterPassword(e.target.value)}
        placeholder="••••••••"
        error={registerErrors.password}
        icon={Lock}
      />
      {registerPassword && renderPasswordStrength()}
      <Input
        label="Xác nhận mật khẩu"
        type="password"
        value={registerConfirmPassword}
        onChange={e => setRegisterConfirmPassword(e.target.value)}
        placeholder="••••••••"
        error={registerErrors.confirm}
        icon={Lock}
      />
      <div className="flex items-start mt-4">
        <div className="flex items-center h-5">
          <input
            id="terms"
            type="checkbox"
            checked={acceptedTerms}
            onChange={e => setAcceptedTerms(e.target.checked)}
            className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
          />
        </div>
        <div className="ml-3 text-sm">
          <label htmlFor="terms" className="text-gray-700">
            Tôi đồng ý với <a href="#" className="text-blue-600 hover:text-blue-800 hover:underline">Điều khoản</a> và <a href="#" className="text-blue-600 hover:text-blue-800 hover:underline">Chính sách bảo mật</a>
          </label>
          {registerErrors.terms && <p className="mt-1 text-sm text-red-500">{registerErrors.terms}</p>}
        </div>
      </div>
      <Button type="submit" disabled={isLoading} className="w-full mt-6">
        {isLoading ? 'Đang xử lý...' : 'Đăng ký'}
      </Button>
      <div className="mt-6 text-center">
        <p className="text-gray-600">
          Đã có tài khoản?{' '}
          <button
            type="button"
            onClick={() => setActiveForm('login')}
            className="text-blue-600 hover:text-blue-800 hover:underline font-medium focus:outline-none"
          >
            Đăng nhập
          </button>
        </p>
      </div>
      {renderSocialButtons()}
    </form>
  );

  const renderForgotPasswordForm = () => (
    <form onSubmit={handleForgotPassword} className="space-y-5 animate-fadeIn">
      <div className="text-center mb-6">
        <div className="mx-auto w-12 h-12 rounded-full bg-blue-100 flex items-center justify-center mb-4">
          <KeyRound size={24} className="text-blue-600" />
        </div>
        <h2 className="text-2xl font-bold text-gray-800">Quên mật khẩu?</h2>
        <p className="text-gray-600 mt-1">Vui lòng nhập email để nhận hướng dẫn đặt lại mật khẩu</p>
      </div>
      <Input
        label="Email"
        type="email"
        value={forgotEmail}
        onChange={e => setForgotEmail(e.target.value)}
        placeholder="email@example.com"
        error={forgotErrors.email}
        icon={Mail}
      />
      <Button
        type="submit"
        disabled={isLoading}
        className="w-full mt-2"
        icon={ArrowRight}
      >
        {isLoading ? 'Đang xử lý...' : 'Gửi hướng dẫn đặt lại'}
      </Button>
      <div className="mt-6 text-center">
        <button
          type="button"
          onClick={() => setActiveForm('login')}
          className="text-blue-600 hover:text-blue-800 hover:underline font-medium focus:outline-none"
        >
          Quay lại đăng nhập
        </button>
      </div>
    </form>
  );

  const clearNotification = () => {
    setError(null);
    setSuccess(null);
  };

  // CSS để thêm animation fadeIn
  useEffect(() => {
    const style = document.createElement('style');
    style.textContent = `
      @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
      }
      .animate-fadeIn {
        animation: fadeIn 0.3s ease-out forwards;
      }
    `;
    document.head.appendChild(style);
    return () => {
      document.head.removeChild(style);
    };
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-blue-50 flex flex-col px-4 py-6 sm:py-12 sm:px-6 lg:px-8">
      <div className="fixed top-6 left-6 sm:top-8 sm:left-8 z-50">
        <Logo />
      </div>

      <div className="mt-16 sm:mt-24 mx-auto w-full max-w-md">
        <div className="bg-white py-8 px-4 sm:px-6 md:px-8 shadow-lg rounded-xl border border-gray-100">
          {error && (
            <Alert
              type="error"
              message={error}
              onClose={clearNotification}
            />
          )}
          {success && (
            <Alert
              type="success"
              message={success}
              onClose={clearNotification}
            />
          )}

          {activeForm === 'login' && renderLoginForm()}
          {activeForm === 'register' && renderRegisterForm()}
          {activeForm === 'forgot' && renderForgotPasswordForm()}
        </div>
        
        <div className="mt-6 text-center text-sm text-gray-600">
          <p>Bạn gặp vấn đề khi đăng nhập? <a href="#" className="text-blue-600 hover:text-blue-800 hover:underline font-medium">Liên hệ hỗ trợ</a></p>
        </div>
      </div>
    </div>
  );
};

export default AuthPage;