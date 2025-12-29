import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [resultMedia, setResultMedia] = useState(null);
  const [summary, setSummary] = useState({});
  const [loading, setLoading] = useState(false);
  const [fileType, setFileType] = useState('');

  // Cleanup URL untuk mencegah memory leak pada browser
  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  const onFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      setSelectedFile(file);
      setFileType(file.type.split('/')[0]); // Mendeteksi 'image' atau 'video'
      setPreviewUrl(URL.createObjectURL(file));
      setResultMedia(null);
      setSummary({});
    }
  };

  const onUpload = async () => {
    if (!selectedFile) return;
    setLoading(true);

    const formData = new FormData();
    formData.append("file", selectedFile);
    const isVideo = selectedFile.type.startsWith('video');

    try {
      const res = await axios.post(`http://127.0.0.1:8000${isVideo ? "/detect-video" : "/detect"}`,
        formData,
        { responseType: isVideo ? 'blob' : 'json' }
      );

      if (isVideo) {
        // Tampilkan Video
        const videoUrl = URL.createObjectURL(res.data);
        setResultMedia(videoUrl);

        // Baca Tabel dari Header
        const summaryHeader = res.headers['x-summary'];
        if (summaryHeader) setSummary(JSON.parse(summaryHeader));
      } else {
        // Tampilkan Gambar & Tabel dari Body JSON
        setResultMedia(res.data.image);
        setSummary(res.data.summary);
      }
    } catch (err) {
      alert("Gagal memproses file.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <header className="header">
        <h1>LettuceEye AI</h1>
        <p>Advanced Lettuce Disease Detection System</p>
      </header>

      <section className="upload-card">
        <input
          type="file"
          onChange={onFileChange}
          className="file-input"
          accept="image/*,video/*"
        />
        <button onClick={onUpload} disabled={loading} className="btn-detect">
          {loading ? "AI is Scanning Frames..." : "Run AI Inference"}
        </button>
      </section>

      <main className="main-grid">
        {/* PANEL KIRI: INPUT SOURCE */}
        <div className="media-card">
          <div className="card-header">
            <span>SOURCE INPUT</span>
            <span className="file-name-tag">{selectedFile ? selectedFile.name : 'No file'}</span>
          </div>
          <div className="media-box">
            {previewUrl ? (
              fileType === 'video' ? (
                <video src={previewUrl} controls />
              ) : (
                <img src={previewUrl} alt="Original" />
              )
            ) : (
              <span className="text-muted">Waiting for input stream...</span>
            )}
          </div>
        </div>

        {/* PANEL KANAN: AI OUTPUT */}
        <div className="media-card">
          <div className="card-header">
            <span>DETECTION RESULT</span>
            {resultMedia && <span className="badge-green status-badge">COMPLETED</span>}
          </div>
          <div className="media-box">
            {resultMedia ? (
              fileType === 'video' ? (
                <video
                  key={resultMedia}
                  src={resultMedia}
                  controls
                  autoPlay
                />
              ) : (
                <img src={resultMedia} alt="Detected" />
              )
            ) : (
              <div className="loading-placeholder">
                {loading ? (
                  <div className="loader-text">Neural Network Processing...</div>
                ) : (
                  <span className="text-muted">Ready for inference</span>
                )}
              </div>
            )}
          </div>

          {/* TABEL RINGKASAN & TOMBOL DOWNLOAD */}
          {Object.keys(summary).length > 0 && (
            <div className="summary-container">
              <h3>Ringkasan Deteksi</h3>
              <table className="summary-table">
                <thead>
                  <tr>
                    <th>Classification</th>
                    <th>Instances</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(summary).map(([name, count]) => (
                    <tr key={name}>
                      <td>
                        <span className={name.toLowerCase().includes('healthy') ? 'badge-green status-badge' : 'badge-red status-badge'}>
                          {name}
                        </span>
                      </td>
                      <td>{count} {fileType === 'video' ? 'Occurrence(s)' : 'Object(s)'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>

              {/* TOMBOL DOWNLOAD DINAMIS */}
              <div style={{ marginTop: '20px', textAlign: 'center' }}>
                <a
                  href={resultMedia}
                  download={fileType === 'video' ? "detected_lettuce.avi" : "detected_lettuce.jpg"}
                  className="btn-download-link"
                >
                  Download {fileType === 'video' ? "Video" : "Gambar"} Hasil
                </a>
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

export default App;