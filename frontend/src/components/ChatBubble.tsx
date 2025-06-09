type Props = {
  sender: "user" | "bot";
  text: string;
};

export default function ChatBubble({ sender, text }: Props) {
  const isUser = sender === "user";
  return (
    <div className={`p-3 rounded-lg max-w-[70%] ${isUser ? "bg-blue-500 text-white self-end" : "bg-gray-200 self-start"}`}>
      {text}
    </div>
  );
}
