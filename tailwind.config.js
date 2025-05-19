/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class',
  content: [
    "./index.html", 
    "./src/**/*.{js,jsx,ts,tsx,css}",
  ],
  theme: {
    extend: {
      colors: {
        primary: {
          DEFAULT: '#3B82F6', // Blue
          dark: '#2563EB',
          light: '#93C5FD',
        },
        secondary: {
          DEFAULT: '#6366F1', // Indigo
          dark: '#4F46E5',
          light: '#A5B4FC',
        },
        accent: {
          DEFAULT: '#8B5CF6', // Purple
          dark: '#7C3AED',
          light: '#C4B5FD',
        },
        success: {
          DEFAULT: '#10B981', // Green
          dark: '#059669',
          light: '#6EE7B7',
        },
        warning: {
          DEFAULT: '#F59E0B', // Amber
          dark: '#D97706',
          light: '#FCD34D',
        },
        error: {
          DEFAULT: '#EF4444', // Red
          dark: '#DC2626',
          light: '#FCA5A5',
        },
        background: {
          light: '#FFFFFF',
          dark: '#1F2937',
        },
        text: {
          light: '#1F2937',
          dark: '#F9FAFB',
        }
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        heading: ['Poppins', 'sans-serif'],
      },
      fontSize: {
        'xs': '0.75rem',
        'sm': '0.875rem',
        'base': '1rem',
        'lg': '1.125rem',
        'xl': '1.25rem',
        '2xl': '1.5rem',
        '3xl': '1.875rem',
        '4xl': '2.25rem',
      },
    },
  },
  plugins: [],
}