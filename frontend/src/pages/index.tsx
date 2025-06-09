import { useState } from "react";
import ChatBubble from "../components/ChatBubble";
import InputBox from "../components/InputBox";
import { sendMessageToBot } from "../utils/api";

type Message = { sender: "bot" | "user"; text: string };

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([{ sender: "bot", text: "Hello! Ask me something." }]);

  const handleSend = async (userMessage: string) => {
    setMessages((prev) => [...prev, { sender: "user", text: userMessage }]);
    const response = await sendMessageToBot(userMessage);
    setMessages((prev) => [...prev, { sender: "bot", text: response }]);
  };

  return (
    <main className="max-w-2xl mx-auto p-4 flex flex-col gap-3">
      <h1 className="text-2xl font-bold text-center">RAG Chatbot</h1>
      <div className="flex flex-col gap-2">
        {messages.map((msg, i) => (
          <ChatBubble key={i} sender={msg.sender} text={msg.text} />
        ))}
      </div>
      <InputBox onSend={handleSend} />
    </main>
  );
}
