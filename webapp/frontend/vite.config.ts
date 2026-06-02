import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig({
  plugins: [react()],

  server: {
    port: 3000,
    strictPort: true,
    // 개발 환경: Vite → FastAPI 프록시 (CORS 우회)
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        secure: false,
      },
    },
  },

  build: {
    outDir: 'dist',
    sourcemap: false,
    // 청크 분리: vendor / recharts / react-router 분리로 초기 번들 최소화
    rollupOptions: {
      output: {
        manualChunks: {
          'react-vendor'  : ['react', 'react-dom', 'react-router-dom'],
          'query-vendor'  : ['@tanstack/react-query', '@tanstack/react-table'],
          'chart-vendor'  : ['recharts'],
        },
      },
    },
  },

  resolve: {
    // "@/components/..." 절대 경로 alias
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
})
