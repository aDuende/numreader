import { useRef, useState, useEffect } from 'react';
import Swal from 'sweetalert2';

const API = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";

interface ModelItem {
  name: string;
  size: number;
  uploaded: string;
  active: boolean;
}

interface Metrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
}

const AdminPage: React.FC = () => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [pendingFile, setPendingFile] = useState<File | null>(null);
  const [models, setModels] = useState<ModelItem[]>([]);
  const [activeName, setActiveName] = useState<string | null>(null);
  const [datasetStats, setDatasetStats] = useState<Record<string, number>>({});
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [retraining, setRetraining] = useState(false);
  const [retrainResult, setRetrainResult] = useState<string>("");
  const [uploading, setUploading] = useState(false);

  const LABEL_DISPLAY: Record<string, string> = {
    thai_0: "๐", thai_1: "๑", thai_2: "๒",
    thai_3: "๓", thai_4: "๔", thai_5: "๕",
  };

  const fetchStats = async () => {
    try {
      const res = await fetch(`${API}/api/dataset/stats`);
      const data = await res.json();
      setDatasetStats(data.stats ?? {});
    } catch { /* backend not ready */ }
  };

  const fetchModels = async () => {
    try {
      const res = await fetch(`${API}/api/models`);
      const data = await res.json();
      setModels(data.models ?? []);
      setActiveName(data.active ?? null);
    } catch { /* ignore */ }
  };

  const fetchMetrics = async () => {
    try {
      const res = await fetch(`${API}/api/models/metrics`);
      const data = await res.json();
      setMetrics(data.metrics);
    } catch { /* ignore */ }
  };

  useEffect(() => {
    fetchStats();
    fetchModels();
    fetchMetrics();
  }, []);

  const handleRetrain = async () => {
    const confirm = await Swal.fire({
      title: "เทรนโมเดลใหม่?",
      text: "จะใช้ข้อมูลทั้งหมดในโฟลเดอร์ dataset/ ใช้เวลาประมาณ 1-2 นาที",
      icon: "question",
      showCancelButton: true,
      confirmButtonText: "เทรนเลย",
      cancelButtonText: "ยกเลิก",
      confirmButtonColor: "#3b82f6",
    });
    if (!confirm.isConfirmed) return;

    setRetraining(true);
    setRetrainResult("");

    try {
      await fetch(`${API}/api/retrain`, { method: "POST" });
      const poll = setInterval(async () => {
        try {
          const res = await fetch(`${API}/api/retrain/status`);
          const data = await res.json();
          if (!data.running && data.last_result) {
            clearInterval(poll);
            setRetraining(false);
            if (data.last_result.success) {
              setRetrainResult("✅ เทรนสำเร็จ! โมเดลใหม่พร้อมใช้งาน");
              fetchStats();
              fetchModels();
              fetchMetrics();
            } else {
              setRetrainResult("❌ เทรนไม่สำเร็จ: " + data.last_result.error);
            }
          }
        } catch { /* ignore */ }
      }, 3000);
    } catch {
      setRetraining(false);
      setRetrainResult("❌ ไม่สามารถเชื่อมต่อ backend ได้");
    }
  };

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    if (!/\.h5$/i.test(file.name)) {
      Swal.fire({
        title: 'ไฟล์ไม่ถูกต้อง!',
        text: 'รองรับเฉพาะ .h5 (Keras model)',
        icon: 'error',
        confirmButtonColor: '#ef4444',
      });
      event.target.value = '';
      return;
    }
    setPendingFile(file);
  };

  const handleConfirmUpload = async () => {
    if (!pendingFile) return;
    setUploading(true);
    try {
      const formData = new FormData();
      formData.append("file", pendingFile);
      const res = await fetch(`${API}/api/models/upload`, {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (res.ok && data.success) {
        await Swal.fire({
          title: 'อัปโหลดสำเร็จ',
          text: `เพิ่ม ${data.name} ในคลังโมเดลแล้ว`,
          icon: 'success',
          confirmButtonColor: '#3b82f6',
        });
        setPendingFile(null);
        if (fileInputRef.current) fileInputRef.current.value = '';
        fetchModels();
      } else {
        Swal.fire('ผิดพลาด', data.detail ?? 'อัปโหลดไม่สำเร็จ', 'error');
      }
    } catch {
      Swal.fire('ผิดพลาด', 'เชื่อมต่อ backend ไม่ได้', 'error');
    } finally {
      setUploading(false);
    }
  };

  const handleActivate = async (name: string) => {
    const result = await Swal.fire({
      title: 'เปลี่ยนโมเดลใช้งาน?',
      text: `สลับไปใช้โมเดล: ${name}`,
      icon: 'info',
      showCancelButton: true,
      confirmButtonText: 'ยืนยัน',
      confirmButtonColor: '#3b82f6',
    });
    if (!result.isConfirmed) return;
    try {
      const res = await fetch(`${API}/api/models/activate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name }),
      });
      const data = await res.json();
      if (res.ok && data.success) {
        Swal.fire('สำเร็จ!', `ระบบใช้โมเดล ${name} แล้ว`, 'success');
        fetchModels();
        fetchMetrics();
      } else {
        Swal.fire('ผิดพลาด', data.detail ?? 'สลับไม่สำเร็จ', 'error');
      }
    } catch {
      Swal.fire('ผิดพลาด', 'เชื่อมต่อ backend ไม่ได้', 'error');
    }
  };

  const handleDelete = async (name: string) => {
    const result = await Swal.fire({
      title: `ลบ ${name}?`,
      icon: 'warning',
      showCancelButton: true,
      confirmButtonText: 'ลบ',
      confirmButtonColor: '#ef4444',
    });
    if (!result.isConfirmed) return;
    try {
      const res = await fetch(`${API}/api/models/${encodeURIComponent(name)}`, {
        method: 'DELETE',
      });
      if (res.ok) {
        fetchModels();
      } else {
        const data = await res.json();
        Swal.fire('ผิดพลาด', data.detail ?? 'ลบไม่สำเร็จ', 'error');
      }
    } catch {
      Swal.fire('ผิดพลาด', 'เชื่อมต่อ backend ไม่ได้', 'error');
    }
  };

  return (
    <div className="flex flex-col items-center py-12 px-4 min-h-screen bg-gray-50">
      <h1 className="text-3xl font-bold text-gray-800 mb-10">
        ระบบจัดการโมเดล AI
      </h1>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 w-full max-w-6xl">

        {/* Dataset Stats + Retrain */}
        <div className="bg-white p-8 shadow-sm border border-gray-100 rounded-2xl">
          <h2 className="text-xl font-bold mb-6 text-gray-700 flex items-center">
            <span className="mr-2">📊</span>ข้อมูล Dataset
          </h2>

          <div className="grid grid-cols-3 gap-3 mb-6">
            {Object.entries(LABEL_DISPLAY).map(([key, thai]) => (
              <div key={key} className="bg-gray-50 rounded-xl p-3 text-center border border-gray-100">
                <div className="text-2xl font-black text-blue-600">{thai}</div>
                <div className="text-xs text-gray-400 mt-1">{datasetStats[key] ?? 0} รูป</div>
              </div>
            ))}
          </div>

          <button
            onClick={fetchStats}
            className="text-xs text-gray-400 hover:text-blue-500 mb-4 underline"
          >
            รีเฟรช
          </button>

          <div className="mt-2 p-4 bg-blue-50 rounded-xl border border-blue-100 text-sm text-blue-700 mb-6">
            <strong>วิธีเก็บข้อมูล:</strong> ไปหน้า User → วาดเลข ๑๐–๑๕ → กด SAVE → บอกว่าวาดเลขอะไร<br/>
            ทำซ้ำหลายๆ ครั้งเพื่อให้ AI เรียนรู้ลายมือของคุณ
          </div>

          <button
            onClick={handleRetrain}
            disabled={retraining}
            className={`w-full py-4 rounded-2xl font-bold text-white transition-all ${
              retraining ? "bg-gray-400 cursor-not-allowed" : "bg-green-500 hover:bg-green-600"
            }`}
          >
            {retraining ? "⏳ กำลังเทรน..." : "🚀 เทรนโมเดลใหม่"}
          </button>

          {retrainResult && (
            <p className="mt-3 text-sm text-center font-bold text-gray-600">{retrainResult}</p>
          )}
        </div>

        {/* Model Management (real upload) */}
        <div className="bg-white p-8 shadow-sm border border-gray-100 rounded-2xl">
          <h2 className="text-xl font-bold mb-2 text-gray-700 flex items-center">
            <span className="mr-2">🧠</span>Model Management
          </h2>
          <p className="text-sm text-gray-500 mb-6">
            Current model:{" "}
            <span className="font-bold text-blue-600">
              {activeName ?? "thai_digit_model.h5 (default)"}
            </span>
          </p>

          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileChange}
            className="hidden"
            accept=".h5"
          />

          <div className="flex gap-3 mb-4">
            <button
              onClick={() => fileInputRef.current?.click()}
              className="flex-1 bg-blue-600 hover:bg-blue-700 text-white py-3 rounded-xl font-bold flex items-center justify-center gap-2"
            >
              📤 Upload Model
            </button>
            <button
              onClick={handleConfirmUpload}
              disabled={!pendingFile || uploading}
              className={`flex-1 py-3 rounded-xl font-bold ${
                !pendingFile || uploading
                  ? "bg-gray-300 text-gray-500 cursor-not-allowed"
                  : "bg-gray-900 hover:bg-black text-white"
              }`}
            >
              {uploading ? "กำลังอัปโหลด..." : "Confirm Upload"}
            </button>
          </div>

          {pendingFile ? (
            <p className="text-sm text-gray-600">
              ไฟล์ที่เลือก: <b>{pendingFile.name}</b>{" "}
              ({(pendingFile.size / 1024 / 1024).toFixed(2)} MB)
            </p>
          ) : (
            <p className="text-sm text-gray-400">ยังไม่ได้อัปโหลด model ใหม่</p>
          )}

          <div className="mt-6 p-3 bg-blue-50 rounded-xl border border-blue-100 text-xs text-blue-700">
            เมื่ออัปโหลดและกด SELECT ในคลังโมเดล ระบบ Predict จะใช้ model ใหม่ทันที (รองรับ .h5)
          </div>

          {metrics && (
            <div className="mt-6 grid grid-cols-2 gap-2 text-center text-xs">
              <div className="bg-gray-50 p-2 rounded-lg border">
                <div className="text-gray-400">Accuracy</div>
                <div className="font-black text-blue-600 text-base">
                  {(metrics.accuracy * 100).toFixed(2)}%
                </div>
              </div>
              <div className="bg-gray-50 p-2 rounded-lg border">
                <div className="text-gray-400">Precision</div>
                <div className="font-black text-blue-600 text-base">
                  {(metrics.precision * 100).toFixed(2)}%
                </div>
              </div>
              <div className="bg-gray-50 p-2 rounded-lg border">
                <div className="text-gray-400">Recall</div>
                <div className="font-black text-blue-600 text-base">
                  {(metrics.recall * 100).toFixed(2)}%
                </div>
              </div>
              <div className="bg-gray-50 p-2 rounded-lg border">
                <div className="text-gray-400">F1-score</div>
                <div className="font-black text-blue-600 text-base">
                  {(metrics.f1 * 100).toFixed(2)}%
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Inventory (real list from backend) */}
        <div className="bg-white p-8 shadow-sm border border-gray-100 rounded-2xl flex flex-col lg:col-span-2">
          <h2 className="text-xl font-bold mb-6 text-gray-700 flex items-center">
            <span className="mr-2">📁</span>คลังโมเดลในระบบ
            <button
              onClick={fetchModels}
              className="ml-auto text-xs text-gray-400 hover:text-blue-500 underline"
            >
              รีเฟรช
            </button>
          </h2>

          {models.length === 0 ? (
            <p className="text-sm text-gray-400 text-center py-6">
              ยังไม่มีโมเดลที่อัปโหลด — เทรนหรืออัปโหลด .h5 เพื่อเริ่มใช้งาน
            </p>
          ) : (
            <div className="space-y-3">
              {models.map((m) => (
                <div
                  key={m.name}
                  className={`p-4 rounded-2xl border transition-all flex items-center justify-between ${
                    m.active
                      ? 'border-blue-500 bg-blue-50 ring-1 ring-blue-500'
                      : 'border-gray-100 bg-white hover:border-gray-300'
                  }`}
                >
                  <div className="flex items-center space-x-4">
                    <div
                      className={`w-10 h-10 rounded-full flex items-center justify-center text-xl ${
                        m.active ? 'bg-blue-500 text-white' : 'bg-gray-100 text-gray-400'
                      }`}
                    >
                      🧠
                    </div>
                    <div>
                      <p
                        className={`font-bold text-sm ${
                          m.active ? 'text-blue-700' : 'text-gray-700'
                        }`}
                      >
                        {m.name}
                      </p>
                      <p className="text-[10px] text-gray-400">
                        อัปโหลด: {m.uploaded} • {(m.size / 1024 / 1024).toFixed(2)} MB
                      </p>
                    </div>
                  </div>

                  <div className="flex items-center gap-2">
                    {m.active ? (
                      <span className="bg-blue-500 text-white text-[10px] px-3 py-1 rounded-full font-black uppercase tracking-wider">
                        Active
                      </span>
                    ) : (
                      <button
                        onClick={() => handleActivate(m.name)}
                        className="text-[10px] font-bold text-gray-500 hover:text-blue-600 hover:bg-white px-3 py-1 rounded-full border border-gray-200"
                      >
                        SELECT
                      </button>
                    )}
                    <button
                      onClick={() => handleDelete(m.name)}
                      disabled={m.active}
                      className={`text-[10px] font-bold px-3 py-1 rounded-full border ${
                        m.active
                          ? 'text-gray-300 border-gray-100 cursor-not-allowed'
                          : 'text-red-500 border-red-200 hover:bg-red-50'
                      }`}
                    >
                      ลบ
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

      </div>
    </div>
  );
};

export default AdminPage;
