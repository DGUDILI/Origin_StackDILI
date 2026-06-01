# ──────────────────────────────────────────────────────────────────────────────
# DGUDILI Frontend — Dockerfile (2-stage: build → nginx serve)
#
# 빌드:
#   docker build --build-arg VITE_API_BASE_URL=https://your-alb.amazonaws.com \
#                -t dgudili-frontend .
# ──────────────────────────────────────────────────────────────────────────────

# ── Stage 1: 빌드 ─────────────────────────────────────────────────────────────
FROM node:20-alpine AS builder

WORKDIR /app

# 의존성 설치 (package.json만 먼저 복사 → layer 캐시 활용)
COPY package.json package-lock.json* ./
RUN npm ci

# 소스 복사 및 빌드
COPY . .

# 빌드 시점에 API URL 주입 (ARG → ENV → Vite 빌드에서 사용)
ARG VITE_API_BASE_URL=""
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL

RUN npm run build

# ── Stage 2: nginx 서빙 ───────────────────────────────────────────────────────
FROM nginx:1.27-alpine

# SPA 라우팅을 위한 nginx 설정 (404 → index.html로 fallback)
RUN rm /etc/nginx/conf.d/default.conf
COPY nginx.conf /etc/nginx/conf.d/default.conf

COPY --from=builder /app/dist /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
