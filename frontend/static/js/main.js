/**
 * Frontend JavaScript for Chatbot App
 */

const chatForm = document.getElementById('chatForm');
const messageInput = document.getElementById('messageInput');
const chatMessages = document.getElementById('chatMessages');
const typingIndicator = document.getElementById('typingIndicator');
const sendBtn = document.getElementById('sendBtn');

let threadId = null;

function appendMessage(content, isUser) {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message');
    messageDiv.classList.add(isUser ? 'user-message' : 'bot-message');
    if (isUser) {
        messageDiv.textContent = content;
    } else {
        messageDiv.innerHTML = marked.parse(content);
    }
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function showTyping() {
    if (typingIndicator) {
        typingIndicator.style.display = 'flex';
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
}

function hideTyping() {
    if (typingIndicator) {
        typingIndicator.style.display = 'none';
    }
}

if (chatForm) {
    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const message = messageInput.value.trim();
        if (!message) return;

        // Add user message
        appendMessage(message, true);
        messageInput.value = '';
        messageInput.disabled = true;
        if (sendBtn) sendBtn.disabled = true;
        showTyping();

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    message: message,
                    thread_id: threadId
                })
            });

            const data = await response.json();

            if (data.success) {
                threadId = data.thread_id;
                appendMessage(data.response, false);
            } else {
                appendMessage("Sorry, something went wrong.", false);
            }
        } catch (error) {
            console.error('Error:', error);
            appendMessage("Sorry, I couldn't connect to the server.", false);
        } finally {
            hideTyping();
            messageInput.disabled = false;
            if (sendBtn) sendBtn.disabled = false;
            messageInput.focus();
        }
    });
}
