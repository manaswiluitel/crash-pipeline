# ---- Build stage ----
FROM golang:1.23-alpine AS builder
WORKDIR /app

# Copy go mod/sum and download deps
COPY go.mod go.sum ./
RUN go mod download

# Copy source and build
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -o extractor .

# ---- Runtime stage ----
FROM alpine:3.20
WORKDIR /app

# Install CA certs for HTTPS (needed for Socrata API) and curl for health checks
RUN apk add --no-cache ca-certificates curl

# Copy binary from builder
COPY --from=builder /app/extractor /app/extractor

# Run the extractor
ENTRYPOINT ["/app/extractor"]
