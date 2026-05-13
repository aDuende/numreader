import { useRef, useState } from "react";
import CanvasDraw from "react-canvas-draw";

const UserPage = () => {
  const canvasRef = useRef<any>(null);
  const [brushSize, setBrushSize] = useState(12);
  const [prediction, setPrediction] = useState<string>("?");
  const [confidence, setConfidence] = useState<number>(0);
  const [warning, setWarning] = useState<string>("");

  const handlePredict = async () => {
    if (!canvasRef.current) return;

    // getDataURL returns a flat PNG of just the drawing on white background.
    // DO NOT use document.querySelector("canvas") — CanvasDraw uses multiple
    // stacked canvas layers and querySelector grabs the grid/background layer.
    const dataUrl: string = canvasRef.current.getDataURL("png", false, "#ffffff");

    const res2 = await fetch(dataUrl);
    const blob = await res2.blob();

    const formData = new FormData();
    formData.append("file", blob, "drawing.png");

    try {
      const res = await fetch(`${import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000"}/api/predict`, {
        method: "POST",
        body: formData,
      });

      const data = await res.json();
      console.log(data);

      const VALID_PREDICTIONS = ["๑๐", "๑๑", "๑๒", "๑๓", "๑๔", "๑๕"];

      if (!data.digits || data.digits.length < 2) {
        // Only 1 or 0 digits detected — user must draw a 2-digit number
        setPrediction("?");
        setConfidence(0);
        setWarning("กรุณาวาดเลข ๑๐ - ๑๕");
        return;
      }

      if (!VALID_PREDICTIONS.includes(data.prediction)) {
        // Predicted number is outside 10-15 range
        setPrediction("?");
        setConfidence(0);
        setWarning("ผลลัพธ์ไม่อยู่ในช่วง ๑๐ - ๑๕ กรุณาวาดใหม่");
        return;
      }

      setWarning("");
      setPrediction(data.prediction);

      // Confidence = units digit confidence only (first digit is hardcoded)
      const unitsConfidence = data.digits[1]?.confidence ?? 0;
      setConfidence(Math.round(unitsConfidence * 100));
    } catch (error) {
      console.error(error);
      alert("AI server error");
    }
  };

  const handleSave = async () => {
    if (!canvasRef.current) return;

    const saveData = canvasRef.current.getSaveData();
    const parsedData = JSON.parse(saveData);

    if (parsedData.lines.length === 0) {
      alert("วาดเลขก่อนแล้วค่อยกด SAVE");
      return;
    }

    // Ask which number was drawn (10-15 only)
    const CHOICES: Record<string, string> = {
      "๑๐": "thai_0", "๑๑": "thai_1", "๑๒": "thai_2",
      "๑๓": "thai_3", "๑๔": "thai_4", "๑๕": "thai_5",
    };
    const chosen = window.prompt(
      "คุณวาดเลขอะไร? พิมพ์ตัวเลข:\n๑๐ / ๑๑ / ๑๒ / ๑๓ / ๑๔ / ๑๕"
    );
    if (!chosen) return;
    const label = CHOICES[chosen.trim()];
    if (!label) {
      alert("กรุณาพิมพ์เลขในช่วง ๑๐ - ๑๕ เท่านั้น");
      return;
    }

    const dataUrl: string = canvasRef.current.getDataURL("png", false, "#ffffff");
    const res = await fetch(dataUrl);
    const blob = await res.blob();

    const formData = new FormData();
    formData.append("file", blob, "drawing.png");
    formData.append("label", label);

    try {
      const response = await fetch(`${import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000"}/api/collect`, {
        method: "POST",
        body: formData,
      });

      if (response.ok) {
        alert(`บันทึก ${chosen} เรียบร้อย! ขอบคุณสำหรับข้อมูล 👍`);
        canvasRef.current.clear();
        setPrediction("?");
        setConfidence(0);
        setWarning("");
      } else {
        alert("เซฟไม่สำเร็จ ลองใหม่");
      }
    } catch {
      alert("ไม่สามารถเชื่อมต่อ backend ได้");
    }
  };

  const handleClear = () => {
    canvasRef.current?.clear();
    setPrediction("?");
    setConfidence(0);
    setWarning("");
  };

  return (
    <div className="flex flex-col items-center py-12 px-4">
      <div className="text-center mb-10">
        <h1 className="text-3xl font-bold text-gray-800 mb-2">
          ทำนายเลขไทย
        </h1>
        <p className="text-gray-500 font-medium">
          วาดเลขไทย ๑๐ - ๑๕ แล้วดูผลลัพธ์ทางขวา
        </p>
      </div>

      <div className="flex flex-col lg:flex-row items-center lg:items-start justify-center gap-8 w-full max-w-6xl">
        <div className="flex flex-col items-center bg-white p-6 rounded-3xl shadow-sm border border-gray-100 w-48">
          <span className="text-xs font-black text-gray-400 uppercase tracking-widest mb-6">
            Tools
          </span>

          <div className="mb-8 flex flex-col items-center">
            <div className="text-[10px] text-gray-400 font-bold uppercase mb-3">
              Preview
            </div>

            <div className="w-20 h-20 bg-gray-50 rounded-xl flex items-center justify-center border border-gray-100 overflow-hidden">
              <div
                className="bg-black rounded-full transition-all duration-75"
                style={{
                  width: `${brushSize * 2}px`,
                  height: `${brushSize * 2}px`,
                  maxWidth: "70px",
                  maxHeight: "70px",
                }}
              />
            </div>

            <span className="text-blue-600 font-bold mt-2 text-sm">
              {brushSize}px
            </span>
          </div>

          <div className="flex flex-col items-center justify-center flex-1 w-full relative">
            <div className="h-32 flex items-center justify-center">
              <input
                type="range"
                min="2"
                max="30"
                value={brushSize}
                onChange={(e) => setBrushSize(parseInt(e.target.value))}
                className="w-32 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-600 -rotate-90 origin-center"
              />
            </div>

            <label className="text-[10px] font-bold text-gray-400 uppercase mt-4">
              Size
            </label>
          </div>
        </div>

        <div className="flex flex-col items-center">
          <div className="bg-white shadow-2xl rounded-[2.5rem] overflow-hidden border-white ring-1 ring-gray-100">
            <CanvasDraw
              ref={canvasRef}
              canvasWidth={360}
              canvasHeight={360}
              brushRadius={brushSize}
              lazyRadius={0}
              brushColor="#000000"
              backgroundColor="#FFFFFF"
              className="cursor-crosshair"
              hideInterface={true}
            />
          </div>

          <div className="mt-8 flex space-x-4 w-full">
            <button
              onClick={handlePredict}
              className="flex-1 bg-blue-600 hover:bg-blue-700 text-white py-4 rounded-2xl font-bold"
            >
              ทำนายผล
            </button>

            <button
              onClick={handleClear}
              className="flex-1 bg-white border-2 border-gray-100 hover:bg-gray-50 text-gray-500 py-4 rounded-2xl font-bold"
            >
              ล้างหน้าจอ
            </button>

            <button
              onClick={handleSave}
              className="flex-1 bg-red-500 hover:bg-red-600 text-white py-4 rounded-2xl font-bold"
            >
              SAVE
            </button>
          </div>
        </div>

        <div className="flex flex-col items-center justify-center">
          <div className="p-10 bg-white rounded-[2.5rem] shadow-sm border border-gray-100 text-center w-64 aspect-square flex flex-col items-center justify-center overflow-hidden">
            <p className="text-xs text-gray-400 uppercase tracking-widest font-black mb-4">
              AI Prediction
            </p>

            {warning ? (
              <div className="flex flex-col items-center justify-center gap-3">
                <span className="text-4xl">✏️</span>
                <p className="text-red-400 font-bold text-sm text-center leading-snug">
                  {warning}
                </p>
              </div>
            ) : (
              <>
                <div
                  className="font-black text-blue-600 break-all leading-none max-w-full text-center"
                  style={{
                    fontSize:
                      prediction.length <= 1
                        ? "7rem"
                        : prediction.length <= 3
                        ? "4rem"
                        : prediction.length <= 6
                        ? "2.5rem"
                        : "1.25rem",
                  }}
                >
                  {prediction}
                </div>

                <div className="mt-6 w-full">
                  <div className="w-full bg-gray-100 rounded-full h-4 overflow-hidden border border-gray-200">
                    <div
                      className="bg-blue-500 h-full rounded-full transition-all duration-700 ease-out"
                      style={{ width: `${confidence}%` }}
                    ></div>
                  </div>
                  <p className="text-gray-500 mt-2 text-xs font-bold uppercase tracking-wider text-center">
                    Confidence: {confidence}%
                  </p>
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default UserPage;