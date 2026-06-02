/** @type {import('tailwindcss').Config} */
export default {
  content: [
    './index.html',
    './src/**/*.{js,ts,jsx,tsx}',
  ],
  theme: {
    extend: {
      colors: {
        // DILI 위험 등급 전용 색상
        risk: {
          high:       '#ef4444',  // red-500
          'high-bg':  '#fef2f2',  // red-50
          'high-ring':'#fca5a5',  // red-300
          low:        '#22c55e',  // green-500
          'low-bg':   '#f0fdf4',  // green-50
          'low-ring': '#86efac',  // green-300
        },
      },
      animation: {
        'progress-indeterminate': 'progress-indeterminate 1.5s ease-in-out infinite',
        'fade-in': 'fade-in 0.3s ease-out',
      },
      keyframes: {
        'progress-indeterminate': {
          '0%'  : { transform: 'translateX(-100%)' },
          '100%': { transform: 'translateX(400%)' },
        },
        'fade-in': {
          '0%'  : { opacity: '0', transform: 'translateY(4px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
      },
    },
  },
  plugins: [],
}
