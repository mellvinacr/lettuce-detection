import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';

import Home from './pages/Home';
import ScanPhoto from './pages/ScanPhoto';
import ScanVideo from './pages/ScanVideo'; // Tambahkan baris ini!
import ScanRealTime from './pages/ScanRealTime.jsx';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/photo" element={<ScanPhoto />} />
        <Route path="/video" element={<ScanVideo />} />
        <Route path="/scan-realtime" element={<ScanRealTime />} />
      </Routes>
    </Router>
  );
}

export default App;