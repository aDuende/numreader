import { useEffect, useMemo, useState } from "react";

const API_BASE = "http://127.0.0.1:8000";

// Single-digit classes used by the model (index → display label).
// The classifier only predicts the units digit (๐-๕); the app frames it as
// the number 10-15 by always prepending ๑.
const CLASS_LABELS = ["๑๐", "๑๑", "๑๒", "๑๓", "๑๔", "๑๕"];
const CLASS_ARABIC = ["10", "11", "12", "13", "14", "15"];

type Metrics = {
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  confusion_matrix: number[][];
};

type MisItem = {
  file: string;
  actual: string;
  predicted: string;
  confidence: number;
  image: string;
};

const ErrorAnalysis = () => {
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [misItems, setMisItems] = useState<MisItem[]>([]);
  const [misTotal, setMisTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string>("");

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [mRes, eRes] = await Promise.all([
          fetch(`${API_BASE}/api/models/metrics`),
          fetch(`${API_BASE}/api/models/misclassified?limit=12`),
        ]);
        if (!mRes.ok) throw new Error(`metrics: ${mRes.status}`);
        if (!eRes.ok) throw new Error(`misclassified: ${eRes.status}`);
        const mJson = await mRes.json();
        const eJson = await eRes.json();
        if (cancelled) return;
        setMetrics(mJson.metrics ?? null);
        setMisItems(eJson.items ?? []);
        setMisTotal(eJson.total ?? (eJson.items?.length ?? 0));
      } catch (e: any) {
        if (!cancelled) setError(e?.message ?? "load failed");
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  // Convert confusion matrix counts to row-normalized percentages.
  const cmPercent = useMemo(() => {
    if (!metrics) return null;
    return metrics.confusion_matrix.map((row) => {
      const total = row.reduce((a, b) => a + b, 0) || 1;
      return row.map((v) => (v / total) * 100);
    });
  }, [metrics]);

  // Identify the most-confused pair (largest off-diagonal cell).
  const topConfusion = useMemo(() => {
    if (!metrics) return null;
    let best = { i: -1, j: -1, count: 0 };
    metrics.confusion_matrix.forEach((row, i) =>
      row.forEach((v, j) => {
        if (i !== j && v > best.count) best = { i, j, count: v };
      })
    );
    return best.count > 0 ? best : null;
  }, [metrics]);

  // Class with lowest per-class accuracy (weakest class).
  const weakestClass = useMemo(() => {
    if (!metrics) return null;
    let worst = { i: -1, acc: 1.1 };
    metrics.confusion_matrix.forEach((row, i) => {
      const total = row.reduce((a, b) => a + b, 0) || 1;
      const acc = row[i] / total;
      if (acc < worst.acc) worst = { i, acc };
    });
    return worst.i >= 0 ? worst : null;
  }, [metrics]);

  // Map raw single-digit prediction (๐..๕) back to its class index.
  const digitToIndex = (d: string) => {
    const i = ["๐", "๑", "๒", "๓", "๔", "๕"].indexOf(d);
    return i >= 0 ? i : -1;
  };

  return (
    <div className="p-8 bg-gray-50 min-h-screen">
      <h1 className="text-3xl font-bold text-gray-800 mb-2">🔍 Error Analysis</h1>
      {metrics && (
        <p className="text-sm text-gray-500 mb-6">
          Accuracy {(metrics.accuracy * 100).toFixed(1)}% &middot; Precision{" "}
          {(metrics.precision * 100).toFixed(1)}% &middot; Recall{" "}
          {(metrics.recall * 100).toFixed(1)}% &middot; F1{" "}
          {(metrics.f1 * 100).toFixed(1)}%
        </p>
      )}
      {error && (
        <div className="mb-6 p-4 bg-red-50 border border-red-100 text-red-600 rounded-2xl text-sm">
          โหลดข้อมูลไม่สำเร็จ: {error} (ตรวจสอบว่า backend ทำงานที่ {API_BASE})
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Confusion Matrix */}
        <div className="bg-white p-6 rounded-3xl shadow-sm border border-gray-100">
          <h2 className="text-xl font-bold mb-6 text-gray-700">Confusion Matrix</h2>
          {loading && !metrics ? (
            <p className="text-sm text-gray-400">กำลังโหลด…</p>
          ) : !cmPercent ? (
            <p className="text-sm text-gray-400">ยังไม่มีข้อมูล metrics</p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-center border-collapse">
                <thead>
                  <tr>
                    <th className="p-2"></th>
                    {CLASS_LABELS.map((c) => (
                      <th
                        key={c}
                        className="p-2 text-xs text-gray-400 font-black"
                      >
                        ทาย {c}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {cmPercent.map((row, rowIndex) => (
                    <tr key={rowIndex}>
                      <td className="p-2 text-xs font-black text-gray-400">
                        จริง {CLASS_LABELS[rowIndex]}
                      </td>
                      {row.map((value, colIndex) => {
                        const isCorrect = rowIndex === colIndex;
                        const display =
                          value >= 10
                            ? `${Math.round(value)}%`
                            : value > 0
                            ? `${value.toFixed(1)}%`
                            : "0%";
                        const rawCount =
                          metrics?.confusion_matrix[rowIndex][colIndex] ?? 0;
                        return (
                          <td
                            key={colIndex}
                            title={`true=${CLASS_LABELS[rowIndex]} pred=${CLASS_LABELS[colIndex]} count=${rawCount}`}
                            className={`p-4 border border-gray-50 text-sm font-bold ${
                              isCorrect
                                ? "bg-green-50 text-green-600"
                                : value > 2
                                ? "bg-red-50 text-red-500"
                                : "text-gray-300"
                            }`}
                          >
                            {display}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          <p className="mt-4 text-xs text-gray-400 italic">
            * ค่า % คำนวณตามแถว (เปอร์เซ็นต์ของตัวอย่างจริงในแต่ละชั้น)
          </p>
        </div>

        {/* Misclassified examples */}
        <div className="bg-white p-6 rounded-3xl shadow-sm border border-gray-100">
          <h2 className="text-xl font-bold mb-2 text-gray-700">
            Misclassified Examples
          </h2>
          <p className="text-xs text-gray-400 mb-4">
            {loading
              ? "กำลังโหลด…"
              : misTotal === 0
              ? "ไม่มีตัวอย่างที่ทายผิดในชุดทดสอบ"
              : `พบที่ทายผิดทั้งหมด ${misTotal} ภาพ — แสดง ${misItems.length} ภาพแรก (ความมั่นใจสูงสุด)`}
          </p>
          <div className="space-y-4 max-h-[480px] overflow-y-auto pr-1">
            {misItems.map((item) => {
              const actualIdx = digitToIndex(item.actual);
              const predIdx = digitToIndex(item.predicted);
              const actualLabel =
                actualIdx >= 0 ? CLASS_LABELS[actualIdx] : `๑${item.actual}`;
              const predLabel =
                predIdx >= 0 ? CLASS_LABELS[predIdx] : `๑${item.predicted}`;
              return (
                <div
                  key={item.file}
                  className="flex items-center p-4 bg-gray-50 rounded-2xl border border-gray-100"
                >
                  <div className="w-16 h-16 bg-white rounded-lg border border-gray-200 flex items-center justify-center overflow-hidden">
                    <img
                      src={item.image}
                      alt={item.file}
                      className="w-full h-full object-contain"
                    />
                  </div>
                  <div className="ml-6 flex-1">
                    <div className="flex justify-between items-center">
                      <span className="text-sm font-bold text-gray-600">
                        จริง:{" "}
                        <span className="text-green-600">{actualLabel}</span>
                      </span>
                      <span className="text-sm font-bold text-gray-600">
                        AI ทาย:{" "}
                        <span className="text-red-500">{predLabel}</span>
                      </span>
                    </div>
                    <div className="mt-2 w-full bg-gray-200 rounded-full h-1.5">
                      <div
                        className="bg-red-400 h-full rounded-full"
                        style={{ width: `${item.confidence}%` }}
                      ></div>
                    </div>
                    <p className="text-[10px] text-gray-400 mt-1 uppercase font-bold">
                      Confidence: {item.confidence}%
                    </p>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Analysis summary derived from real data */}
      <div className="mt-8 bg-blue-600 p-8 rounded-[2.5rem] text-white shadow-xl shadow-blue-200">
        <h2 className="text-xl font-bold mb-4">
          💡 บทสรุปการวิเคราะห์ (Analysis Summary)
        </h2>
        <ul className="list-disc list-inside space-y-2 text-blue-100 font-medium">
          {metrics && (
            <li>
              ความแม่นยำรวมของโมเดล (Accuracy) ={" "}
              <b>{(metrics.accuracy * 100).toFixed(1)}%</b> บนชุดทดสอบ
            </li>
          )}
          {topConfusion && (
            <li>
              โมเดลสับสนระหว่างเลข{" "}
              <b>
                {CLASS_LABELS[topConfusion.i]} ({CLASS_ARABIC[topConfusion.i]})
              </b>{" "}
              กับ{" "}
              <b>
                {CLASS_LABELS[topConfusion.j]} ({CLASS_ARABIC[topConfusion.j]})
              </b>{" "}
              มากที่สุด ({topConfusion.count} ภาพ)
            </li>
          )}
          {weakestClass && (
            <li>
              เลขที่โมเดลทำได้แย่ที่สุดคือ{" "}
              <b>
                {CLASS_LABELS[weakestClass.i]} (
                {CLASS_ARABIC[weakestClass.i]})
              </b>{" "}
              — accuracy ต่อชั้น {(weakestClass.acc * 100).toFixed(1)}%
            </li>
          )}
          {weakestClass && (
            <li>
              <b>ข้อเสนอแนะ:</b> ควรเพิ่ม Data ของเลข{" "}
              <b>{CLASS_LABELS[weakestClass.i]}</b> ในชุดฝึกสอน (Training Set)
              ให้มากขึ้น
            </li>
          )}
          {!metrics && !loading && (
            <li>ยังไม่มี metrics.json — โปรดเทรนโมเดลด้วย train_test_model.py</li>
          )}
        </ul>
      </div>
    </div>
  );
};

export default ErrorAnalysis;
