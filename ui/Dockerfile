# syntax=docker/dockerfile:1.7

# UI container — Next.js static export served by nginx.

ARG UI_BUILD_IMAGE=node:24.22.1-alpine3.24@sha256:6a1c190e17215d844926b0677f4bf87273ff7d5e54db06d3b5b5863ab0b09a3d
ARG NGINX_VERSION=1.31.5-alpine3.24@sha256:34f40471dea485273c5e2a04dd5e97a682332ceb4a9adecd67de450dcb2fb390

# ---------- builder ----------
FROM ${UI_BUILD_IMAGE} AS builder

ENV NEXT_TELEMETRY_DISABLED=1 \
    npm_config_fund=false \
    npm_config_audit=false

WORKDIR /app

# Layer the lockfile-only install above the source copy so source-only
# edits don't bust the install cache.
COPY ui/litellm-dashboard/package.json ui/litellm-dashboard/package-lock.json ./
RUN --mount=type=cache,target=/root/.npm \
    npm ci --prefer-offline

COPY ui/litellm-dashboard/ ./
RUN npm run build

# ---------- runtime ----------
FROM nginx:${NGINX_VERSION} AS runtime

# Drop the upstream default :80 server; we own the config.
RUN rm -f /etc/nginx/conf.d/default.conf

# Static export → web root.
COPY --from=builder /app/out /usr/share/nginx/html

# Routing rules — see ui/nginx.conf for the full description.
COPY ui/nginx.conf /etc/nginx/nginx.conf

EXPOSE 3000/tcp

# nginx as PID 1 in foreground; respects SIGTERM out of the box, so
# no tini/dumb-init wrapper needed.
CMD ["nginx", "-g", "daemon off;"]
