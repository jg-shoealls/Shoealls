/** @type {import('next').NextConfig} */
const nextConfig = {
  // API 프록시: 개발 시 CORS 없이 FastAPI 호출
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${process.env.API_URL ?? "http://localhost:8000"}/api/:path*`,
      },
    ];
  },
};

module.exports = nextConfig;
