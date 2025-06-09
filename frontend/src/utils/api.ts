export async function sendMessageToBot(message: string): Promise<string> {
  const res = await fetch("http://localhost:8000/chat", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ message })
  });

  const data = await res.json();
  return data.response || "Error communicating with backend.";
}
