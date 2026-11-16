const form = document.querySelector("#message-form");
const input = document.querySelector("#message-input");
const sendButton = document.querySelector("#send-button");
const conversation = document.querySelector("#conversation");
const suggestions = document.querySelector("#suggestions");
const history = [];

function currentTime() {
  return new Intl.DateTimeFormat([], {
    hour: "numeric",
    minute: "2-digit",
  }).format(new Date());
}

function addMessage(role, text) {
  const row = document.createElement("article");
  row.className = `message-row ${role === "user" ? "user-row" : "assistant-row"}`;

  const avatar = document.createElement("div");
  avatar.className = "message-avatar";
  avatar.setAttribute("aria-hidden", "true");
  avatar.textContent = role === "user" ? "Y" : "M";

  const bubble = document.createElement("div");
  bubble.className = `message-bubble ${role === "user" ? "user-bubble" : "assistant-bubble"}`;

  const paragraph = document.createElement("p");
  paragraph.textContent = text;
  const time = document.createElement("time");
  time.textContent = currentTime();

  bubble.append(paragraph, time);
  row.append(avatar, bubble);
  conversation.append(row);
  conversation.scrollTop = conversation.scrollHeight;
}

function showTyping() {
  const row = document.createElement("article");
  row.className = "message-row assistant-row";
  row.id = "typing-indicator";

  const avatar = document.createElement("div");
  avatar.className = "message-avatar";
  avatar.setAttribute("aria-hidden", "true");
  avatar.textContent = "M";

  const bubble = document.createElement("div");
  bubble.className = "message-bubble assistant-bubble";
  bubble.setAttribute("aria-label", "MediAi is typing");
  const dots = document.createElement("span");
  dots.className = "typing-indicator";
  dots.innerHTML = "<span></span><span></span><span></span>";
  bubble.append(dots);
  row.append(avatar, bubble);
  conversation.append(row);
  conversation.scrollTop = conversation.scrollHeight;
}

async function sendMessage(text) {
  const message = text.trim();
  if (!message || sendButton.disabled) {
    return;
  }

  suggestions.hidden = true;
  addMessage("user", message);
  const previousHistory = history.slice(-8);
  history.push({ role: "user", content: message });
  input.value = "";
  input.style.height = "auto";
  input.disabled = true;
  sendButton.disabled = true;
  showTyping();

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, history: previousHistory }),
    });
    const result = await response.json();
    if (!response.ok) {
      throw new Error(result.error || "The request could not be completed.");
    }

    addMessage("assistant", result.answer);
    history.push({ role: "assistant", content: result.answer });
  } catch (error) {
    const friendlyMessage = error instanceof Error
      ? error.message
      : "Please check your connection and try again.";
    addMessage("assistant", `I couldn't get a response. ${friendlyMessage}`);
  } finally {
    document.querySelector("#typing-indicator")?.remove();
    input.disabled = false;
    sendButton.disabled = false;
    input.focus();
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  sendMessage(input.value);
});

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 120)}px`;
});

document.querySelectorAll(".suggestion").forEach((button) => {
  button.addEventListener("click", () => sendMessage(button.textContent));
});
