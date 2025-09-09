import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import './index.css';

function App() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [question, setQuestion] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [report, setReport] = useState(null);
  const [error, setError] = useState(null);
  const [dragOver, setDragOver] = useState(false);

  // 处理图片选择
  const handleImageSelect = (file) => {
    if (file && file.type.startsWith('image/')) {
      setSelectedImage(file);
      const reader = new FileReader();
      reader.onload = (e) => {
        setImagePreview(e.target.result);
      };
      reader.readAsDataURL(file);
      setError(null);
    } else {
      setError('请选择有效的图片文件');
    }
  };

  // 处理文件输入变化
  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      handleImageSelect(file);
    }
  };

  // 处理拖拽
  const handleDragOver = (e) => {
    e.preventDefault();
    setDragOver(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setDragOver(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) {
      handleImageSelect(file);
    }
  };

  // 处理点击上传区域
  const handleUploadAreaClick = () => {
    document.getElementById('fileInput').click();
  };

  // 流式处理鉴赏请求
  const handleSubmit = async () => {
    if (!selectedImage && !question.trim()) {
      setError('请上传图片或输入问题');
      return;
    }

    setIsLoading(true);
    setError(null);
    setReport(''); // 初始化空报告用于流式更新

    try {
      let url = '/api/v1/appraisal/stream';
      let requestOptions = {
        method: 'POST'
      };
      
      if (selectedImage) {
        // 图片+文本的混合输入
        const formData = new FormData();
        formData.append('image', selectedImage);
        if (question.trim()) {
          formData.append('question', question);
        }
        requestOptions.body = formData;
      } else {
        // 纯文本问题
        requestOptions.headers = {
          'Content-Type': 'application/json',
        };
        requestOptions.body = JSON.stringify({
          question: question
        });
      }

      // 发起流式请求，添加超时控制
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 60000); // 60秒超时
      
      requestOptions.signal = controller.signal;
      
      const response = await fetch(url, requestOptions);
      clearTimeout(timeoutId); // 清除超时定时器
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      // 处理流式响应
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      
      while (true) {
        const { done, value } = await reader.read();
        
        if (done) {
          break;
        }
        
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop(); // 保留不完整的行
        
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6).trim();
            
            if (data === '[DONE]') {
              setIsLoading(false);
              return;
            }
            
            if (data === '[START]') {
              continue;
            }
            
            try {
              const parsed = JSON.parse(data);
              if (parsed.type === 'content' && parsed.text) {
                setReport(prev => (prev || '') + parsed.text);
              } else if (parsed.type === 'structured_report' && parsed.data) {
                // 处理结构化报告
                setReport(prev => {
                  const currentReport = prev || '';
                  const structuredData = parsed.data;
                  
                  // 生成结构化显示内容
                  let structuredContent = '\n\n---\n\n## 📋 详细鉴宝报告\n\n';
                  
                  if (structuredData.appraisal_report) {
                    const report = structuredData.appraisal_report;
                    
                    structuredContent += `### 🏺 物品信息\n`;
                    structuredContent += `- 名称: ${report.item_name || '待确定'}\n`;
                    structuredContent += `- 类别: ${report.category || '未知'}\n`;
                    structuredContent += `- 朝代: ${report.dynasty || '待确定'}\n`;
                    structuredContent += `- 材质: ${report.material || '待分析'}\n\n`;
                    
                    if (report.authenticity) {
                      structuredContent += `### 🔍 真伪鉴定\n`;
                      structuredContent += `- 真伪评分: ${report.authenticity.score || 0}/100\n`;
                      structuredContent += `- 可信度: ${report.authenticity.confidence || '待评估'}\n`;
                      structuredContent += `- 分析: ${report.authenticity.analysis || '需要进一步分析'}\n\n`;
                    }
                    
                    if (report.value_estimation) {
                      structuredContent += `### 💰 价值评估\n`;
                      structuredContent += `- 市场价值: ${report.value_estimation.market_value || '待评估'}\n`;
                      structuredContent += `- 收藏价值: ${report.value_estimation.collection_value || '待评估'}\n`;
                      if (report.value_estimation.factors && report.value_estimation.factors.length > 0) {
                        structuredContent += `- 影响因素: ${report.value_estimation.factors.join(', ')}\n`;
                      }
                      structuredContent += '\n';
                    }
                    
                    if (report.condition) {
                      structuredContent += `### 🔧 保存状况\n`;
                      structuredContent += `- 整体状况: ${report.condition.overall || '未检测'}\n`;
                      structuredContent += `- 详细描述: ${report.condition.details || '需要详细检查'}\n\n`;
                    }
                    
                    if (report.historical_context) {
                      structuredContent += `### 📚 历史背景\n`;
                      structuredContent += `${report.historical_context}\n\n`;
                    }
                    
                    if (report.recommendations && report.recommendations.length > 0) {
                      structuredContent += `### 💡 专业建议\n`;
                      report.recommendations.forEach((rec, index) => {
                        structuredContent += `${index + 1}. ${rec}\n`;
                      });
                    }
                  }
                  
                  return currentReport + structuredContent;
                });
              } else if (parsed.chunk) {
                // 兼容旧格式
                setReport(prev => (prev || '') + parsed.chunk);
              }
            } catch (e) {
              // 如果不是JSON，直接作为文本块处理
              if (data && data !== '') {
                setReport(prev => (prev || '') + data);
              }
            }
          }
        }
      }
    } catch (err) {
      console.error('鉴赏请求失败:', err);
      if (err.name === 'AbortError') {
        setError('请求超时，请检查网络连接或稍后重试');
      } else {
        setError(`鉴赏失败: ${err.message}`);
      }
    } finally {
      setIsLoading(false);
    }
  };

  // 重置表单
  const handleReset = () => {
    setSelectedImage(null);
    setImagePreview(null);
    setQuestion('');
    setReport(null);
    setError(null);
    document.getElementById('fileInput').value = '';
  };

  return (
    <div className="container">
      <div className="card">
        <h1 className="title">🏺 古董鉴赏系统</h1>
        
        {error && (
          <div className="error-message">
            {error}
          </div>
        )}

        {/* 图片上传区域 */}
        <div 
          className={`upload-area ${dragOver ? 'dragover' : ''}`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={handleUploadAreaClick}
        >
          <input
            id="fileInput"
            type="file"
            accept="image/*"
            onChange={handleFileChange}
            style={{ display: 'none' }}
          />
          
          {imagePreview ? (
            <div>
              <img src={imagePreview} alt="预览" className="preview-image" />
              <p className="upload-hint">点击重新选择图片</p>
            </div>
          ) : (
            <div>
              <div className="upload-text">📸 点击或拖拽上传古董图片</div>
              <div className="upload-hint">支持 JPG、PNG、GIF 格式</div>
            </div>
          )}
        </div>

        {/* 问题输入区域 */}
        <div className="input-group">
          <label className="input-label" htmlFor="questionInput">
            💭 您想了解什么？
          </label>
          <textarea
            id="questionInput"
            className="text-input"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="请输入您的问题，例如：这是什么朝代的瓷器？有什么特点？"
            disabled={isLoading}
          />
        </div>

        {/* 操作按钮 */}
        <div style={{ display: 'flex', gap: '10px' }}>
          <button
            className="submit-btn"
            onClick={handleSubmit}
            disabled={isLoading || (!selectedImage && !question.trim())}
            style={{ flex: 1 }}
          >
            {isLoading ? '🔍 AI鉴赏中...' : '🚀 开始鉴赏'}
          </button>
          
          {(selectedImage || question || report) && (
            <button
              className="submit-btn"
              onClick={handleReset}
              disabled={isLoading}
              style={{ 
                flex: '0 0 120px',
                background: 'linear-gradient(135deg, #718096 0%, #4a5568 100%)'
              }}
            >
              🔄 重新开始
            </button>
          )}
        </div>
      </div>

      {/* 加载状态 */}
      {isLoading && (
        <div className="card">
          <div className="loading-container">
            <div className="loading-spinner"></div>
            <div className="loading-text">AI正在仔细分析您的古董，请稍候...</div>
          </div>
        </div>
      )}

      {/* 鉴赏报告 */}
      {report && (
        <div className="card">
          <div className="report-container">
            <h2 className="report-title">
              📋 鉴赏报告
            </h2>
            <div className="report-content">
              <ReactMarkdown>{report || '暂无分析结果'}</ReactMarkdown>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;