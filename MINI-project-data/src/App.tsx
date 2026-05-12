import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import UserPage from './UserPage';
import AdminPage from './AdminPage';
import ErrorAnalysis from './ErrorAnalysis';

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-[#F9FAFB] font-sans">
        <nav className="bg-white/80 backdrop-blur-md sticky top-0 z-10 border-b border-gray-100">
          <div className="max-w-4xl mx-auto px-6 py-4 flex justify-between items-center">
            <span className="font-black text-xl tracking-tight text-blue-600">THAI DIGIT AI</span>
            <div className="space-x-8">
              <Link to="/" className="text-gray-600 hover:text-blue-600 font-medium transition-colors">หน้าวาดรูป</Link>
              <Link to="/admin" className="text-gray-600 hover:text-blue-600 font-medium transition-colors">Admin</Link>
              <Link to="/error" className="text-gray-600 hover:text-blue-600 font-medium transition-colors">Error Analysis</Link>
            </div>
          </div>
        </nav>

        {/* เนื้อหาแต่ละหน้า */}
        <main className="max-w-5xl mx-auto">
          <Routes>
            <Route path="/" element={<UserPage />} />
            <Route path="/admin" element={<AdminPage />} />
            <Route path="/error" element={<ErrorAnalysis />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;