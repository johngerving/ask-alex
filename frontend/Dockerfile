FROM node:20-slim as builder

WORKDIR /app

# Install dependencies
COPY package.json yarn.lock ./
RUN yarn install --frozen-lockfile

# Copy source code
COPY . .
COPY .env.public .env

# Build the application
ENV NODE_ENV=production
RUN yarn build

# Production stage
FROM node:20-slim

WORKDIR /app

# Copy built assets from builder
COPY --from=builder /app/build ./build
COPY --from=builder /app/package.json ./package.json
COPY --from=builder /app/node_modules ./node_modules

# Expose the port the app runs on
EXPOSE 3000

# Start the application
ENV NODE_ENV=production
ENV PORT=3000
CMD ["node", "build"] 