import { useState } from "react";

export default function InputBox({ onSend }: { onSend: (text: string) => void }) {
  const [input, setInput] = useState("");

  const handleSubmit = () => {
    if (!input.trim()) return;
    onSend(input);
    setInput("");
  };

  return (
    <div className="flex gap-2">
      <input
        type="text"
        value={input}
        onChange={(e) => setInput(e.target.value)}
        className="flex-grow border p-2 rounded"
        placeholder="Ask something..."
      />
      <button onClick={handleSubmit} className="bg-blue-600 text-white px-4 rounded">
        Send
      </button>
    </div>
  );
}
