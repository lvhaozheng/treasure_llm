const { createProxyMiddleware } = require('http-proxy-middleware');

module.exports = function(app) {
  // 代理 /api 路径
  app.use(
    '/api',
    createProxyMiddleware({
      target: 'http://127.0.0.1:5000',
      changeOrigin: true,
      logLevel: 'debug',
      timeout: 60000,
      proxyTimeout: 60000,
      agent: false,
      selfHandleResponse: false,
      buffer: false,
      headers: {
        'Connection': 'keep-alive'
      },
      onError: (err, req, res) => {
        console.log('Proxy error:', err);
        res.writeHead(500, {
          'Content-Type': 'text/plain'
        });
        res.end('Proxy error: ' + err.message);
      },
      onProxyReq: (proxyReq, req, res) => {
        console.log('Proxying request:', req.method, req.url);
      },
      onProxyRes: (proxyRes, req, res) => {
        console.log('Proxy response:', proxyRes.statusCode, req.url);
      }
    })
  );
  
  // 代理 /health 路径
  app.use(
    '/health',
    createProxyMiddleware({
      target: 'http://127.0.0.1:5000',
      changeOrigin: true,
      logLevel: 'debug',
      timeout: 60000,
      proxyTimeout: 60000,
      agent: false,
      headers: {
        'Connection': 'keep-alive'
      },
      onError: (err, req, res) => {
        console.log('Proxy error:', err);
        res.writeHead(500, {
          'Content-Type': 'text/plain'
        });
        res.end('Proxy error: ' + err.message);
      },
      onProxyReq: (proxyReq, req, res) => {
        console.log('Proxying request:', req.method, req.url);
      },
      onProxyRes: (proxyRes, req, res) => {
        console.log('Proxy response:', proxyRes.statusCode, req.url);
      }
    })
  );
};